# %%
import mmap
from os import isatty
import numpy as np

"""
from parse_logfile import TextLog
with TextLog('/mnt/gs/departmentN5/preprocessing/example_data/2021_01_14-14_59_35_GrassyLandscape_Cont.log') as log:
    start_trial_times = log.parse_all_state_times(state='StartTrial', times='StateStarted')
    end_trial_times = log.parse_all_state_times(state='EndTrial', times='StateStarted')
print(start_trial_times)
print(end_trial_times)

"""


""" This module allows the logfile to be parsed

"""

class TextLog:
    """
    input:
        path to text version of continuous log file

    """
    sep = b','
    ts_col = 0
    id_col = 1
    logtypes_col = 2

    name_type = ('ObjectName', 'StateMachineName', 'StateName') # Order of this tuple is the type category
    ObjectName = 0 # Index in name_type tuple
    StateMachineName = 1
    StateName = 2

    # Measured timing parameters
    frame_rate = 1/0.01668 # according to IDXGISwapChain
    # projector delay https://www.projectorcentral.com/canon-wux450st-projector-review.htm
    delay = 0.018 #((0.019+0.023)/2) #s
    rendering_frames = 2 # (5 original cameras -> FisheyeBourkeBigCam-> DomeCam)

    #name_cols = [b'LogTypes::ObjectName', b'LogTypes::StateMachineName', b'LogTypes::StateName']

    def __init__(self, path_to_logfile, encoding='UTF-8'):
        self.path = path_to_logfile
        self.enc = encoding


    def __enter__(self):
        self.f = open(self.path, 'rb')
        try:
            self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            self.f.close()
            raise(e)

        return self


    def __exit__(self, type, value, traceback):
        # handle any exception
        self.mm.close()
        self.f.close()


    def goto_start_of_line(self, ind, st=0):
        start_line = self.mm.rfind(b'\n', st, ind)+1
        self.mm.seek(start_line)


    def split_xyz(self, xyz):
        # expects input like b'X=-27.015 Y=-1.476 Z=2912.699'
        xyz_list = xyz.split(b' ')
        out = []
        for dim in xyz_list:
            try:
                tmp = float(dim[2:])
            except ValueError:
                print('Could not convert to float so replacing with nan:', dim)
                tmp = np.nan
            out.append(tmp)
        return np.asarray(out)


    def split_params(self, header_line, param_line):

        out = dict()

        header_cols = header_line.split(self.sep)
        header_type = header_cols[self.logtypes_col][:-6] # Excluding "Header"
        #for param_name in header_cols[self.logtypes_col+1:]:
        #    if param_name in parameters:
        #        raise ValueError('Code not ready to handle duplicate initial parameter %s', param_name.decode(self.enc))

        value_cols = param_line.split(self.sep)
        if header_type != value_cols[self.logtypes_col]:
            raise ValueError('Param line has unexpected logtype %s', value_cols[self.logtypes_col].decode(self.enc))

        string_func = lambda x: x.decode(self.enc).rstrip()

        if header_type == b"LogTypes::FloatParameter":
            func = float
        elif header_type == b"LogTypes::IntParameter":
            func = int
        elif header_type == b"LogTypes::BoolParameter":
            func = lambda x: bool(int(x))
        elif header_type == b"LogTypes::StringParameter":
            func = string_func
        elif header_type == b"LogTypes::VectorParameter":
            func = self.split_xyz
        else:
            print('Unknown Parameter type, data will be stored as string')
            func = string_func

        for name, val in zip(header_cols[self.logtypes_col+1:], value_cols[self.logtypes_col+1:]):
            out[string_func(name)] = func(val)

        return out


    def read_first_line(self):
        # read in the timestamp line
        self.mm.seek(0)
        info = self.mm.readline()  # prints b"Hello Python!\n"
        info = info.split()

        self.level = info[0].decode(self.enc)
        self.start_time = info[-1].decode(self.enc)


    def read_log_header(self):
        header = dict()

        self.read_first_line()

        header['Level'] = self.level
        header['Start Time'] = self.start_time

        # read in other lines a dictionary
        reading = True
        while reading:
            line = self.mm.readline().decode(self.enc).strip()

            if len(line) == 0:
                reading = False
                break

            if line.startswith('0'):
                reading = False
                break
            
            params = line.split(sep=':')
            if len(params) == 2:
                header[params[0]] = params[1].strip()
            elif len(params) == 4:
                sub_params = params[2].split(',')
                header[params[0] + params[1]] = sub_params[0].strip()
                header[params[0] + sub_params[1]] = params[3].strip()
            else:
                print('Unknown number of params %i'%(len(params)))
        self.header = header
        return header


    def calc_nframes(self):
        self.goto_start_of_line(-1, st=0)
        line = self.mm.readline()
        cols = line.split(self.sep)
        final_time = float(cols[self.ts_col])

        self.n_frames = int(final_time*self.frame_rate)


    def make_id_struct(self):
        tp = np.dtype([
            ('id', 'uint32'),
            ('id_repeat', 'uint32'),
            ('id_type', 'uint8'),
            ('name', object), # this is basically a list, a bunch of pointers
            ('start', 'uint32'),
            ('end', 'int32')])

        self.all_ids = np.zeros(10000, dtype=tp)

        name_cols = [bytes('LogTypes::%s'%(iname),self.enc)  for iname in self.name_type]
        search_bytes = b',LogTypes::GUID,'

        st = 0
        i_id = 0
        reading = True
        while reading:
            ind = self.mm.find(search_bytes, st)
            if ind == -1:
                reading = False
                break
            self.goto_start_of_line(ind)
            line = self.mm.readline()
            cols = line.split(self.sep)
            this_id = cols[self.id_col]
            this_id = int(this_id[1:-1]) # remove []
            id_match = self.all_ids['id'] == this_id

            if np.any(id_match):
                # end of previous
                final_ind = np.max(np.nonzero(id_match))
                self.all_ids[final_ind]['end'] = ind-1
                # count repeats
                self.all_ids[i_id]['id_repeat'] = np.sum(id_match, dtype=int)

            self.all_ids[i_id]['id'] = this_id

            # Get name line
            line = self.mm.readline()
            cols = line.split(self.sep)

            # parse naming line
            next_id = cols[self.id_col]
            next_id = int(next_id[1:-1])
            logtype = cols[self.logtypes_col]
            name = cols[-1]

            if this_id != next_id:
                raise ValueError('For %i next line has wrong object or state id %i',
                    this_id, next_id)
            elif logtype not in name_cols:
                raise ValueError('Unexpected name log type %s for %i',
                    logtype.decode(self.enc), this_id)

            match_ind = []
            for ii, itype in enumerate(name_cols):
                if itype in logtype:
                    match_ind.append(ii)

            if not match_ind:
                raise ValueError('Unknown logtype %s for %i',
                    logtype.decode(self.enc), this_id)


            self.all_ids[i_id]['name'] = name.decode(self.enc).rstrip()
            self.all_ids[i_id]['id_type'] = match_ind[0]
            self.all_ids[i_id]['start'] = ind

            i_id += 1
            if i_id == self.all_ids.shape[0]:
                print('Increased all_ids size')
                self.all_ids = np.concatenate((self.all_ids, np.zeros(1000, dtype=tp)))
            st = ind + len(search_bytes)

        # trim to correct size
        self.all_ids = self.all_ids[:i_id]

        # find final time
        # do we need the true final time here? Only useful if we're trying to speed up the location search
        self.all_ids['end'][self.all_ids['end'] == 0] = -1

    def find_player_id(self, obj='AnimalCharacter'):
        # create structure and check it
        if not hasattr(self, 'all_ids'):
            self.make_id_struct()

        # player spawns always near the start
        for id in self.all_ids:
            if id["name"].startswith(obj):
                ids = id["id"]
                st = id["start"]
                end = id["end"]
                break
        return ids, st, end

    def find_object_id(self, obj):
        # create structure and check it
        if not hasattr(self, 'all_ids'):
            self.make_id_struct()

        inc = self.all_ids["id_type"] == self.ObjectName

        isname = self.all_ids[inc]["name"] == obj
        ids = self.all_ids[inc][isname]["id"]
        st = self.all_ids[inc][isname]["start"]
        end = self.all_ids[inc][isname]["end"]
        return ids, st, end


    def find_state_ids(self, state, starts_with=False):
        # create structure and check it
        if not hasattr(self, 'all_ids'):
            self.make_id_struct()

        inc = np.nonzero(self.all_ids["id_type"] == self.StateName)[0]

        isname = self.all_ids[inc]["name"] == state + '_C_0'  # _C_0 to ignore duplicates

        if np.any(isname):
            print('Old log file version')
        else:
            isname = np.zeros(inc.shape, dtype=bool)
            for ii, i_id in enumerate(self.all_ids[inc]):
                if state+'_C_' in i_id['name']:
                    isname[ii] = True

        ids = self.all_ids[inc[isname]]["id"]
        st = self.all_ids[inc[isname]]["start"]
        end = self.all_ids[inc[isname]]["end"]


        return ids, st, end


    def find_id_st_end(self, this_id):
        # create structure and check it
        if not hasattr(self, 'all_ids'):
            self.make_id_struct()

        isid = self.all_ids["id"] == this_id
        st = self.all_ids[isid]["start"]
        end = self.all_ids[isid]["end"]
        return st, end


    def find_qpf(self):
        # find query performance frequency
        if not hasattr(self, 'header'):
            self.read_log_header()

        try:
            qpf = self.header['Query Performance Frequency']
        except KeyError:
            print('Key "Query Performance Frequency " not found in header, 10000000 used')
            qpf = 10000000

        self.qpf = float(qpf)


    def make_screen_times(self):
        
        self.fake_times = False
        
        def fill_nan(A):
            '''
            interpolate to fill nan values
            '''
            inds = np.arange(A.shape[0])
            good = np.isfinite(A)
            if np.all(good):
                return A
            else:
                A[~good] = np.interp(inds[~good], inds[good], A[good])
                return A


        def interpolate_swap_stats(counts, times, cued):
            counts = counts.astype(int)
            all_frame_counts =  np.unique(counts)
            
            min_frame = all_frame_counts.min()
            max_frame = all_frame_counts.max()

            # make array as if we caught all frames
            frame_counts = np.arange(min_frame, max_frame+1, dtype=int)
            frame_times = np.full(frame_counts.shape, np.nan)
            frame_cued = np.full(frame_counts.shape, np.nan)
            
            frame_times[counts-min_frame] = times
            frame_cued[counts-min_frame] = cued

            # find missed frame times and interpolate
            missed_frames = np.nonzero(np.isnan(frame_times))[0]
            frame_times = fill_nan(frame_times)
            print('N frames missed:', missed_frames.size)
            #print(missed_frames)
            
            return frame_counts, frame_times, frame_cued


        def fake_screen_times():
            # round each timestamp down to the nearest multiple of refreshrate
            # throw warning that skipped frames need to be found from the photodiode
            
            self.fake_times = True
            self.fake_delay = 0.085 #s
            print('Unknown screen timing, use photodiode to find correct times')
            
            # character input is on every tick unless the game is paused 
            self.make_id_struct()
            for name in self.all_ids['name']:
                if name.startswith('AnimalCharacter'):
                    break
            unused, ts_input = self.parse_input(obj=name,convert=False)
            
            log_evt, log_ts_evt, evt_desc, log_qpc_evt = self.parse_eventmarkers(convert=False)
            
            refresh = 1/self.frame_rate
            if not hasattr(self, 'qpf'):
                self.find_qpf()
            self.first_swap_qpc = 0
            
            if np.sum(log_qpc_evt) == 0:
                print('No qpc times for eventmarkers, using rounded log times instead')

                inter_tick_interval = np.insert(np.diff(ts_input), 0, ts_input[0])
                log_frame_n = np.rint(inter_tick_interval/refresh)
                frame_ts = np.cumsum(log_frame_n)*refresh
                
            else:
                # preallocate
                frame_ts = np.full(ts_input.shape,np.nan)

                evt_brightness = 996
                inc = log_evt == evt_brightness
                brightness_qpc = log_qpc_evt[inc]
                self.first_swap_qpc = log_qpc_evt[0]
                
                # convert to seconds
                brightness_qpc = (brightness_qpc - self.first_swap_qpc)/self.qpf

                # insert in correct log position
                pos = np.searchsorted(ts_input, log_ts_evt[inc])
                frame_ts[pos] = brightness_qpc

                # fill the non_photodiode times
                for ii in np.setxor1d(np.arange(ts_input.size),pos):
                    tick_qpc = log_qpc_evt[ts_input[ii] == log_ts_evt]
                    if tick_qpc.size == 0:
                        continue
                    frame_ts[ii] = (tick_qpc[0] - self.first_swap_qpc)/self.qpf # use closest to start of the tick
            
            self.log_to_screen_times = np.concatenate( 
                (ts_input[:,np.newaxis], frame_ts[:,np.newaxis]+self.fake_delay), 
                axis=1)
            
            return
            

        framestats, ts_stats = self.parse_viewportframe_stats()
        
        if ts_stats.size <= 1:
            print('No viewport frame statistics found, using eventmarker qpc')
            fake_screen_times()
            return
        
        if np.any(framestats.sum(axis=0) == 0):
            print('Some viewport frame statistics missing, using eventmarker qpc')
            fake_screen_times()
            return

        SyncQPCTime = 0
        PresentCount = 1
        ThisPresentId = 2
        GFrameCounter = 5
        GFrameNumber = 6

        # find valid swap times
        inc = framestats[:,SyncQPCTime]>0

        frame_counts, frame_times, frame_cued = interpolate_swap_stats(
            framestats[inc,PresentCount],
            framestats[inc,SyncQPCTime],
            framestats[inc,ThisPresentId] - framestats[inc,PresentCount])

        # convert to seconds
        self.first_swap_qpc = frame_times[0]
        frame_times -= self.first_swap_qpc
        if not hasattr(self, 'qpf'):
            self.find_qpf()
        frame_times /= self.qpf #s

        # Adjust frame_times based on projector delay
        frame_times += self.delay

        # Make sure GFrameNumber looks like we expect
        if (framestats[0,PresentCount] - framestats[0,GFrameNumber]) != 0:
            if framestats[5,SyncQPCTime] == 0:
                print('Something went wrong with viewport frame statistics, using eventmarker qpc')
                fake_screen_times()
                return
            else:
                #assert False, 'GFrameNumber does not match first PresentCount, code is not built to handle this'
                print('GFrameNumber does not match first PresentCount, automatically adjusting')
                adj = framestats[0,PresentCount] - framestats[0,GFrameNumber]
                framestats[:,GFrameNumber] = framestats[:,GFrameNumber] + adj
        else: # that check passed, time for next check
            assert np.all((framestats[:,GFrameNumber] - framestats[:,GFrameCounter]) == np.ones(1)), 'GFrameNumber is not exactly 1 more than GFrameCounter, code is not build to handle this'

        # let's assign each game thread time to a frame number
        # GFrameNumber should be the times sent to render https://docs.unrealengine.com/4.27/en-US/API/Runtime/Core/GFrameNumber/
        thread_frame = (framestats[:,GFrameNumber] + self.rendering_frames).astype(int) 

        # remove times that are too long
        min_frame = frame_counts[0]
        max_frame = frame_counts[-1]

        inc = thread_frame < max_frame
        thread_frame = thread_frame[inc]

        # this is the true n_frames
        self.n_frames = thread_frame.size

        # preallocate 
        self.log_to_screen_times = np.zeros((self.n_frames,2))

        self.log_to_screen_times[:,0] = ts_stats[inc]
        self.log_to_screen_times[:,1] = frame_times[thread_frame - min_frame]
        
    
    def convert_log_to_screen(self, ts):
        # grab the start byte of every frame here so we can parse per frame later?
        
        if not hasattr(self, 'log_to_screen_times'):
            self.make_screen_times()

        #log_to_screen_times should always be a sorted array
        pos = np.searchsorted(self.log_to_screen_times[:,0], ts)

        # handle any log ts from before/after the last flips
        n_frames = self.log_to_screen_times.shape[0]
        inc = np.logical_and(pos < n_frames, pos > 0)

        if np.all(inc):
            screen_time = self.log_to_screen_times[pos,1]
        elif inc.size == 1:
            screen_time = np.nan
        else:
            screen_time = np.full((ts.size), np.nan)
            screen_time[inc] = self.log_to_screen_times[pos[inc],1]
            
        return screen_time
        

    #def find_obj_timestamps(self):
        # if we grabbed the start of each object reference that might help to find just that object
        # most objects only live for the length of a trial
        # it should be the spawn time of every object

    def parse_stateid_times(self, state_id, st, end, times='StateStarted', return_index=False, convert=True):
        zero_buff = 1000 # trials per block
        ts = np.zeros((zero_buff))
        if return_index:
            index = np.zeros((zero_buff), dtype=int)

        search_bytes =  bytes('[%i],LogTypes::%s'%(state_id, times), self.enc)
        #search_bytes =  bytes('['+istate+'],LogTypes::'+times, self.enc)  #[150360],LogTypes::StateStarted

        self.mm.seek(st)
        itime = 0
        reading = True
        while reading:
            ind = self.mm.find(search_bytes, st, end)
            if ind == -1:
                reading = False
                break
            self.goto_start_of_line(ind)
            line = self.mm.readline()
            cols = line.split(self.sep)
            ts[itime] = float(cols[self.ts_col])
            if return_index:
                index[itime] = ind
            itime += 1
            if itime == ts.size:
                print('Had to increase array size')
                ts = np.concatenate((ts, np.zeros((zero_buff))))
                if return_index:
                    index = np.concatenate((index, np.zeros((zero_buff), dtype=int)))
            st = ind + len(search_bytes)

        # trim
        ts = ts[:itime]
        
        if convert:
            ts = self.convert_log_to_screen(ts)

        if return_index:
            #trim
            index = index[:itime]
            return ts, index
        else:
            return ts


    def parse_all_state_times(self, state='StartTrial', times='StateStarted', return_index=False):
        # all_ts is a list with one numpy array per ? (block?)

        state_ids, starts, ends = self.find_state_ids(state)

        all_ts = []
        all_index = []


        for ii, istate in enumerate(state_ids):
            if return_index:
                tmp = self.parse_stateid_times(istate, starts[ii], ends[ii], times=times, return_index=return_index)
                all_ts.append(tmp[0])
                all_index.append(tmp[1])
            else:
                tmp = self.parse_stateid_times(istate, starts[ii], ends[ii], times=times, return_index=return_index)
                all_ts.append(tmp)

        if return_index:
            return all_ts, all_index
        else:
            return all_ts


    # def parse_state_changes(self, state_id, st, end):
    #     zero_buf = 100


    def parse_initial_parameters(self, obj_id=None, st=None, end=None, obj=None):
        # Check for that id ONLY during the initial frame.
        """
        0.000000,[161885],LogTypes::ObjectName,Flickerpattern
        0.000000,[161885],LogTypes::SpawnLocation,X=-159460.063 Y=8470.843 Z=139.244
        0.000000,[161885],LogTypes::SpawnRotation,X=7.259 Y=57.412 Z=-91.747
        0.000000,[161885],LogTypes::FloatParameterHeader,CurrentBrightness
        0.000000,[161885],LogTypes::FloatParameter,1.000000
        0.000000,[161885],LogTypes::VectorParameterHeader,CurrentFlickerColor
        0.000000,[161885],LogTypes::VectorParameter,X=1.000 Y=1.000 Z=1.000
        """

        if obj_id is None:
            if obj is None:
                raise ValueError('Neither obj_id nor obj were set')

            obj_id, st, end = self.find_object_id(obj)

            if len(obj_id) == 0:
                raise ValueError('Object named %s was not found', obj)
            elif len(obj_id) > 1:
                raise ValueError('Object %s found with multiple ids %i',
                    obj, len(obj_id))
            obj_id = obj_id[0]
            st = st[0]
            end = end[0]

        if st is None:
            raise ValueError('Object st must be given to parse the initial parameters')

        search_bytes = bytes('[%i]'%(obj_id), self.enc)

        # find the initial timestamp
        iobj = self.mm.find(search_bytes, st, end) #[92935]
        self.goto_start_of_line(iobj, st=st)
        line = self.mm.readline()
        cols = line.split(self.sep)
        init_ts = cols[self.ts_col] # 0.000000
        search_bytes = init_ts + self.sep + search_bytes

        # create parameter dictionary
        if obj is None:
            isid = np.logical_and(self.all_ids["id"] == obj_id, self.all_ids["start"] == st)
            obj = self.all_ids[isid]["name"][0]

        parameters = dict()
        parameters['ObjectName'] = obj
        parameters['SpawnLogTime'] = float(init_ts)
        parameters['SpawnTime'] = self.convert_log_to_screen(np.array(float(init_ts)))
        parameters['ObjectID'] = obj_id

        # find all params
        reading = True
        while reading:
            iobj = self.mm.find(search_bytes, st, end) #0.000000,[161885]
            if iobj == -1:
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read object line
            line = self.mm.readline()
            if b"SpawnLocation" in line:
                cols = line.split(self.sep)
                parameters['SpawnLocation'] = self.split_xyz(cols[-1])

            elif b"SpawnRotation" in line:
                cols = line.split(self.sep)
                parameters['SpawnRotation'] = self.split_xyz(cols[-1])

            elif b"ParameterHeader" in line:
                header_cols = line.split(self.sep)
                header_type = header_cols[self.logtypes_col][:-6] # Excluding "Header"
                for param_name in header_cols[self.logtypes_col+1:]:
                    if param_name in parameters:
                        raise ValueError('Code not ready to handle duplicate initial parameter %s', param_name.decode(self.enc))

                line = self.mm.readline()
                #print(line)
                value_cols = line.split(self.sep)
                if header_type != value_cols[self.logtypes_col]:
                    raise ValueError('Next line has unexpected logtype %s', value_cols[self.logtypes_col].decode(self.enc))

                string_func = lambda x: x.decode(self.enc).rstrip()

                if header_type == b"LogTypes::FloatParameter":
                    func = float
                elif header_type == b"LogTypes::IntParameter":
                    func = int
                elif header_type == b"LogTypes::BoolParameter":
                    func = lambda x: bool(int(x))
                elif header_type == b"LogTypes::StringParameter":
                    func = string_func
                elif header_type == b"LogTypes::VectorParameter":
                    func = self.split_xyz
                else:
                    print('Unknown Parameter type, data will be stored as string')
                    func = string_func

                for name, val in zip(header_cols[self.logtypes_col+1:], value_cols[self.logtypes_col+1:]):
                    parameters[string_func(name)] = func(val)

        return parameters


    def parse_parameters(self, obj_id=None, st=0, end =-1, obj=None):
        # Check for that id throughout file (once per frame?).
        # Can update end here for future use
        """
        4.637419,[161885],LogTypes::FloatParameterHeader,CurrentBrightness
        4.637419,[161885],LogTypes::FloatParameter,0.000000
        """
        # Create 1 dictionary of all parameters for values and a second dictionary for timestamps
        #param['CurrentBrightness'] = [0,0.5,1,0.5,0,...]
        #ts['CurrentBrightness'] = [0,0.01,0.02,0.03,0.04,...]

        # if not hasattr(self, 'n_frames'):
        #     self.calc_nframes()
        #self.n_frames

        # find the initial parameters
        if obj_id is None:
            if obj is None:
                raise ValueError('Neither obj_id nor obj were set')

            param = self.parse_initial_parameters(obj)
            # find search area st and end
            obj_id, st_obj, end_obj = self.find_object_id(obj)

            if len(obj_id) == 0:
                raise ValueError('Object named %s was not found', obj)
            elif len(obj_id) > 1:
                raise ValueError('Object %s found with multiple ids %i',
                    obj, len(obj_id))

            if st_obj[0] > st:
                st = st_obj[0]
            if end_obj[0] < end:
                if end_obj[0] != -1:
                    end = end_obj[0]
        else:
            if st == 0:
                raise ValueError('st must be given with obj_id')
            param = self.parse_initial_parameters(obj_id=obj_id,st=st,end=end)

        ts = param['SpawnTime']
        param_ts = {x: [ts] for x in param} # new timestamp dictionary

        search_bytes = bytes('[%i]'%(obj_id), self.enc)

        reading = True
        while reading:
            iobj = self.mm.find(search_bytes, st, end) #0.000000,[161885]
            if iobj == -1:
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read object line
            line = self.mm.readline()
            if b"ParameterHeader" in line:
                #print(line)
                cols = line.split(self.sep)
                ts = float(cols[self.ts_col])
                if ts == param['SpawnLogTime']:
                    continue # skip initialization frames

                param_line = self.mm.readline()
                tmp_dict = self.split_params(line, param_line)

                for param_name in tmp_dict:
                    if param_name in param:
                        param_ts[param_name].append(self.convert_log_to_screen(ts))
                        try:
                            param[param_name].append(tmp_dict[param_name])
                        except AttributeError: # not a list
                            param[param_name] = [param[param_name]] + [tmp_dict[param_name]]
                            

                    else:
                        print('Uninitialized paramater %s found'%(param_name))
                        param[param_name] = tmp_dict[param_name]
                        param_ts[param_name] = [self.convert_log_to_screen(ts)]
                        
        # make time stamps np arrays
        for param_name in param_ts:
            param_ts[param_name] = np.asarray(param_ts[param_name])

        # if end == -1:
        #     # This is not safe, doesn't check if it's the only id
        #     self.all_ids[self.all_ids['id'] == obj_id]['end'] = st

        return param, param_ts


    def parse_position(self, obj_id=None, st=0, end =-1, obj=None, convert=True):

        if obj_id is None:
            if obj is None:
                raise ValueError('Neither obj_id nor obj were set')

            obj_id, st, end = self.find_object_id(obj)
            if len(obj_id) == 0:
                raise ValueError('Object named %s was not found', obj)
            elif len(obj_id) > 1:
                raise ValueError('Object %s found with multiple ids %i',
                    obj, len(obj_id))

            st = st[0]
            end = end[0]
            obj_id = obj_id[0]
        
        search_bytes = bytes('[%i]'%(obj_id), self.enc)

        # preallocate
        if not hasattr(self, 'n_frames'):
            self.calc_nframes()

        zero_buff = self.n_frames # 1000=16s
        location = np.zeros((zero_buff,3))
        rotation = np.zeros((zero_buff,3))
        timestamps = np.zeros((zero_buff))

        iloc = 0
        irot = 0
        reading = True
        while reading:
            #line = self.mm.readline()
            iobj = self.mm.find(search_bytes, st, end) #[92935]
            if iobj == -1:
                # could use st to update end here
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read in full line
            line = self.mm.readline()
            #print(line)

            if b"LogTypes::Location" in line:
                # 4.531385,[92819],LogTypes::Location,X=72.614 Y=19.017 Z=585.354
                cols = line.split(self.sep)
                locs = self.split_xyz(cols[-1])
                ts = float(cols[self.ts_col])
                #print(ts, locs)

                location[iloc, :] = locs
                iloc += 1
            elif b"LogTypes::Rotation" in line:
                # 4.531385,[92819],LogTypes::Rotation,X=0.000 Y=10.000 Z=20.000
                cols = line.split(self.sep)
                rots = self.split_xyz(cols[-1])
                ts = float(cols[self.ts_col])
                #print(ts, rots)

                rotation[irot, :] = rots
                irot +=1
            else:
                continue

            # figure out timestamps
            if iloc > irot:
                timestamps[iloc-1] = ts
            elif irot > iloc:
                timestamps[irot-1] = ts
            elif iloc == irot:
                if timestamps[iloc-1] != ts:
                    raise ValueError('Location %i and Rotation %i timestamps %f and %f are not aligned'%(iloc, irot, timestamps[iloc], ts))

            # increase array size if necessary
            if max(iloc, irot) >= timestamps.size:
                print('Had to increase array size')
                location = np.concatenate((location, np.zeros((zero_buff,3))))
                rotation = np.concatenate((rotation, np.zeros((zero_buff,3))))
                timestamps = np.concatenate((timestamps, np.zeros((zero_buff))))

        if iloc != irot:
            print('Different number of Locations and Rotations found')
            trim_ind = max(iloc,irot)
        else:
            trim_ind = iloc

        if convert:
            timestamps = self.convert_log_to_screen(timestamps[:trim_ind])

        return location[:trim_ind,:], rotation[:trim_ind,:], timestamps[:trim_ind]


    def parse_spherical(self, obj_id=None, st=0, end =-1, obj=None, convert=True):

        if obj_id is None:
            if obj is None:
                raise ValueError('Neither obj_id nor obj were set')

            obj_id, st, end = self.find_object_id(obj)
            if len(obj_id) == 0:
                raise ValueError('Object named %s was not found', obj)
            elif len(obj_id) > 1:
                raise ValueError('Object %s found with multiple ids %i',
                    obj, len(obj_id))

            st = st[0]
            end = end[0]
            obj_id = obj_id[0]
        
        search_bytes = bytes('[%i],LogTypes::Spherical'%(obj_id), self.enc)

        # preallocate
        if not hasattr(self, 'n_frames'):
            self.calc_nframes()

        zero_buff = self.n_frames # 1000=16s
        location = np.zeros((zero_buff,3))
        timestamps = np.zeros((zero_buff))

        iloc = 0
        irot = 0
        reading = True
        while reading:
            #line = self.mm.readline()
            iobj = self.mm.find(search_bytes, st, end) #[73108],LogTypes::Spherical
            if iobj == -1:
                # could use st to update end here
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read in full line
            line = self.mm.readline()
            # 176.250412,[73108],LogTypes::Spherical,X=-25.785 Y=-4.053 Z=921.862
            cols = line.split(self.sep)
            locs = self.split_xyz(cols[-1])
            ts = float(cols[self.ts_col])
            #print(ts, locs)

            location[iloc, :] = locs
            iloc += 1

            # figure out timestamps
            timestamps[iloc-1] = ts

            # increase array size if necessary
            if iloc >= timestamps.size:
                print('Had to increase array size')
                location = np.concatenate((location, np.zeros((zero_buff,3))))
                timestamps = np.concatenate((timestamps, np.zeros((zero_buff))))

        trim_ind = iloc

        if convert:
            timestamps = self.convert_log_to_screen(timestamps[:trim_ind])

        return location[:trim_ind,:], timestamps[:trim_ind]


    def parse_eventmarkers(self, st=0, end=-1, zero_buff = 500000, convert=True):
        #19.278986,[74751],LogTypes::Eventmarker,996,153270967478
        #19.278986,[74751],LogTypes::EventmarkerDescription,PhotodiodeUpdate,QPC

        # preallocate
        evt = np.zeros((zero_buff), dtype=np.int64)
        ts = np.zeros((zero_buff))
        evt_desc = [None] * zero_buff
        true_ts = np.zeros((zero_buff), dtype=np.int64)

        # loop through file
        ievt = 0
        reading = True
        parse_bytes = b',LogTypes::Eventmarker,'
        while reading:
            index_evt = self.mm.find(parse_bytes, st, end)
            if index_evt == -1:
                reading = False
                break
            self.goto_start_of_line(index_evt, st=st)
            line = self.mm.readline()
            cols = line.split(self.sep)
            ts[ievt] = float(cols[self.ts_col])

            # loop through all coloumns
            param_st = self.logtypes_col+1
            for ii, param in enumerate(cols[param_st:]):
                if ii == 0:
                    evt[ievt] = int(param)
                elif ii == 1:
                    true_ts[ievt] = int(param)
                else:
                    print('Unexpected parameter in LogTypes::Eventmarker')

            line = self.mm.readline()
            if b"LogTypes::EventmarkerDescription," in line:
                cols = line.split(self.sep)
                evt_desc[ievt] = cols[param_st].decode(self.enc).rstrip()

            ievt += 1
            if ievt == evt.size:
                print('Had to increase array size')
                evt = np.concatenate((evt, np.zeros((zero_buff))))
                ts = np.concatenate((ts, np.zeros((zero_buff))))
                evt_desc.extend([None] * zero_buff)
                true_ts = np.concatenate((true_ts, np.zeros((zero_buff))))

            st = index_evt + len(parse_bytes) # for next find

        if convert:
            ts = self.convert_log_to_screen(ts[:ievt])
            true_ts = (true_ts[:ievt] - self.first_swap_qpc)/self.qpf

        return evt[:ievt], ts[:ievt], evt_desc[:ievt], true_ts[:ievt]


    def parse_photodiode_color(self, obj_id=None, st=0, end =-1, obj=None, convert=True):
        # 4.087554,[133956],LogTypes::PhotodiodeColor,X=1.000 Y=1.000 Z=1.000

        if obj_id is None:
            if obj is None:
                raise ValueError('Neither obj_id nor obj were set')

            obj_id, st, end = self.find_object_id(obj)
            if len(obj_id) == 0:
                raise ValueError('Object named %s was not found', obj)
            elif len(obj_id) > 1:
                raise ValueError('Object %s found with multiple ids %i',
                    obj, len(obj_id))

            st = st[0]
            end = end[0]
            obj_id = obj_id[0]
        
        search_bytes = bytes('[%i],LogTypes::PhotodiodeColor,'%(obj_id), self.enc)

        # preallocate
        if not hasattr(self, 'n_frames'):
            self.calc_nframes()
        zero_buff = self.n_frames # 1000=16s
        colors = np.zeros((zero_buff,3))
        timestamps = np.zeros((zero_buff))
        
        icolors = 0
        reading = True
        while reading:
            #line = self.mm.readline()
            iobj = self.mm.find(search_bytes, st, end) #[92935]
            if iobj == -1:
                # could use st to update end here
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read in full line
            line = self.mm.readline()
            cols = line.split(self.sep)
            color = cols[-3:]

            timestamps[icolors] = float(cols[self.ts_col])
            colors[icolors, :] = np.asarray(color)
            icolors += 1

            # increase array size if necessary
            if icolors >= timestamps.size:
                print('Had to increase array size')
                colors = np.concatenate((colors, np.zeros((zero_buff,3))))
                timestamps = np.concatenate((timestamps, np.zeros((zero_buff))))

        if convert:
            timestamps = self.convert_log_to_screen(timestamps[:icolors])

        return colors[:icolors,:], timestamps[:icolors,:]


    def parse_photodiode_brightness(self, obj_id=None, st=0, end =-1, obj=None, convert=True):
        # 1.342409,[90793],LogTypes::PhotodiodeBrightness,1.000000

        if obj_id is None:
            if obj is None:
                raise ValueError('Neither obj_id nor obj were set')

            obj_id, st, end = self.find_object_id(obj)
            if len(obj_id) == 0:
                raise ValueError('Object named %s was not found', obj)
            elif len(obj_id) > 1:
                raise ValueError('Object %s found with multiple ids %i',
                    obj, len(obj_id))

            st = st[0]
            end = end[0]
            obj_id = obj_id[0]

        search_bytes = bytes('[%i],LogTypes::PhotodiodeBrightness,'%(obj_id), self.enc)

        # preallocate
        if not hasattr(self, 'n_frames'):
            self.calc_nframes()
        zero_buff = self.n_frames # 1000=16s
        brightness = np.zeros((zero_buff))
        timestamps = np.zeros((zero_buff))

        ibrght = 0
        reading = True
        while reading:
            iobj = self.mm.find(search_bytes, st, end) #[92935],LogTypes::PhotodiodeBrightness,
            if iobj == -1:
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read in full line
            line = self.mm.readline()
            
            cols = line.split(self.sep)
            brightness[ibrght] = float(cols[-1])
            timestamps[ibrght] = float(cols[self.ts_col])

            ibrght += 1

            # increase array size if necessary
            if ibrght >= timestamps.size:
                print('Had to increase array size')
                brightness = np.concatenate((brightness, np.zeros((zero_buff))))
                timestamps = np.concatenate((timestamps, np.zeros((zero_buff))))
        
        if convert:
            timestamps = self.convert_log_to_screen(timestamps[:ibrght])

        return brightness[:ibrght], timestamps[:ibrght]

    def parse_viewportframe_stats(self, st=0, end=-1):
        # 0.322909,[70752],LogTypes::ViewportFrameStatistics,56763221960,77625,77627,339617,339617
        # hopefully obj_id is unique so we don't separate
        # Comes from here: https://docs.microsoft.com/en-us/windows/win32/api/dxgi/ns-dxgi-dxgi_frame_statistics
        # And here: https://docs.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgiswapchain-getlastpresentcount 

        search_bytes = bytes(',LogTypes::ViewportFrameStatistics,', self.enc)

        nparam = 7
        """ 
                //SwapChain->GetFrameStatistics(&Stats)
                "%llu,%u,%u,%u,%u,%u,%u"
					Stats.SyncQPCTime.QuadPart,
					Stats.PresentCount,
					ThisPresentId,
					Stats.PresentRefreshCount,
					Stats.SyncRefreshCount,
					GFrameCounter,
					GFrameNumber);
        """ 
        if not hasattr(self, 'n_frames'):
            self.calc_nframes()
        zero_buff = self.n_frames # 1000=16s
        framestats = np.zeros((zero_buff, nparam))
        timestamps = np.zeros((zero_buff))

        ipresent = 0
        reading = True
        while reading:
            iobj = self.mm.find(search_bytes, st, end)
            if iobj == -1:
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read in full line
            line = self.mm.readline()
            cols = line.split(self.sep)

            # check with data this is and fill that in with parameters
            timestamps[ipresent] = float(cols[self.ts_col])

            param_st = self.logtypes_col+1
            if cols[param_st] == b'\r\n':
                continue
            
            # loop through all coloumns
            for ii, param in enumerate(cols[param_st:]):
                # if param == b'EMPTY':
                #     framestats[ipresent,ii] = np.nan
                # else:
                framestats[ipresent,ii] = float(param)

            ipresent += 1

            # increase array size if necessary
            if ipresent >= timestamps.size:
                print('Had to increase array size')
                framestats = np.concatenate((framestats, np.zeros((zero_buff,nparam))))
                timestamps = np.concatenate((timestamps, np.zeros((zero_buff))))

        return framestats[:ipresent,:], timestamps[:ipresent]

    def parse_input(self, obj_id=None, st=0, end =-1, obj=None, convert=True):
        # 1.342409,[90803],LogTypes::InputData,0.000000,0.000000,0.000000

        if obj_id is None:
            if obj is None:
                raise ValueError('Neither obj_id nor obj were set')

            obj_id, st, end = self.find_object_id(obj)
            if len(obj_id) == 0:
                raise ValueError('Object named %s was not found', obj)

            elif len(obj_id) > 1:
                raise ValueError('Object %s found with multiple ids %i',
                    obj, len(obj_id))

            
            st = st[0]
            end = end[0]
            obj_id = obj_id[0]

        search_bytes = bytes('[%i],LogTypes::InputData,'%(obj_id), self.enc)

        # preallocate
        if not hasattr(self, 'n_frames'):
            self.calc_nframes()
        zero_buff = self.n_frames # 1000=16s
        input_data = np.zeros((zero_buff,3))
        timestamps = np.zeros((zero_buff))

        iinput = 0
        reading = True
        while reading:

            iobj = self.mm.find(search_bytes, st, end) #[92935]
            if iobj == -1:
                # could use st to update end here
                reading = False
                break

            self.goto_start_of_line(iobj, st=st)
            st = iobj + len(search_bytes) # for next find

            # read in full line
            line = self.mm.readline()
            cols = line.split(self.sep)
            inpts = cols[-3:]
            timestamps[iinput] = float(cols[self.ts_col])
            input_data[iinput, :] = np.asarray(inpts)
            iinput += 1
            
            # increase array size if necessary
            if iinput >= timestamps.size:
                print('Had to increase array size')
                input_data = np.concatenate((input_data, np.zeros((zero_buff,3))))
                timestamps = np.concatenate((timestamps, np.zeros((zero_buff))))

        if convert:
            timestamps = self.convert_log_to_screen(timestamps[:iinput])

        return input_data[:iinput,:], timestamps[:iinput]

    def parse_message_times(self, message='Overlapped CollisionBox', st=0, end=-1, convert=True):
        
        search_bytes = bytes('],LogTypes::Message,%s'%(message), self.enc)

        # preallocate
        zero_buff = 2000
        timestamps = np.zeros((zero_buff))
        object_id = np.zeros((zero_buff))

        imess = 0
        reading = True
        while reading:

            mess_loc = self.mm.find(search_bytes, st, end)
            if mess_loc == -1:
                reading = False
                break

            self.goto_start_of_line(mess_loc, st=st)
            st = mess_loc + len(search_bytes) # for next find

            # read in full line
            line = self.mm.readline()
            cols = line.split(self.sep)
            timestamps[imess] = float(cols[self.ts_col])
            object_id[imess] = int(cols[self.id_col][1:-1]) # remove []
            imess += 1
            
            # increase array size if necessary
            if imess >= timestamps.size:
                print('Had to increase array size')
                object_id = np.concatenate((object_id, np.zeros((zero_buff))))
                timestamps = np.concatenate((timestamps, np.zeros((zero_buff))))

        if convert:
            timestamps = self.convert_log_to_screen(timestamps[:imess])

        return object_id[:imess], timestamps[:imess]



    def get_ids_per_segment(self, st=0, end=-1):
        # find all ids that were referenced during a segment of the file
        if not hasattr(self, 'all_ids'):
            self.make_id_struct()

        # find eligible ids
        inc = np.logical_or(self.all_ids["end"] > st, self.all_ids["end"] == -1)
        if end != -1:
            inc = np.logical_and(self.all_ids["start"] < end, inc)
        inc = np.nonzero(inc)[0]

        ids = self.all_ids[inc]["id"]
        valid_ids = np.zeros((inc.size), dtype=bool)
        #print(valid_ids.shape)

        for ii, i_id in enumerate(ids):

            search_bytes = bytes('[%i]'%(i_id), self.enc)
            index_id = self.mm.find(search_bytes, st, end)
            if index_id != -1:
                valid_ids[ii] = True

        return self.all_ids[inc][valid_ids]


    # def get_info_per_trial(self, return_spawned=False, spawned_name = 'ImageStimulus', return_param = ['ObjectName', 'Image', 'StimulusCategory'], return_loc = False, return_states=False, return_eventmarkers=False):

    #     st_ts, st_ind = self.parse_all_state_times(state='StartTrial', times='StateStarted', return_index=True)
    #     end_ts, end_ind = self.parse_all_state_times(state='EndTrial', times='StateStarted', return_index=True)

    #     # n_ts = sum([len(x) for x in ts])
    #     # st_ind = np.zeros((n_ts), dtype=int)
    #     # ii = 0
    #     # for times in ts:
    #     #     for itime in times:
    #     #         ts_bytes = b'\n%0.6f,['%(itime) # time stored to 6 decimal places
    #     #         st_ind[ii] = self.mm.find(ts_bytes)

    #     #         ii += 1

    #     all_trials = []
    #     for iblock, index in enumerate(st_ind):
    #         block_trials = []
    #         for iseg, st in enumerate(index):
    #             if iseg >= end_ind[iblock].size:
    #                 # more start trials than end trials
    #                 continue
    #             end = end_ind[iblock][iseg]

    #             trl_ids = self.get_ids_per_segment(st=st, end=end)
    #             trl_states = trl_ids[trl_ids["id_type"] == self.StateName]
    #             trl_obj = trl_ids[trl_ids["id_type"] == self.ObjectName]

    #             # create trial dictionary
    #             trial = dict()
    #             trial['StartTime'] = st_ts[iblock][iseg]
    #             trial['EndTime'] = end_ts[iblock][iseg]

    #             # parsing objects
    #             if return_spawned:
    #                 trial['SpawnedStimuli'] = []
    #                 for ii, i_id in enumerate(trl_obj):
    #                     if spawned_name in i_id['name']:
    #                         obj_id = i_id['id']
    #                         spawn_st = i_id['start']
    #                         if spawn_st >= st and spawn_st < end:
    #                             tmp = self.parse_initial_parameters(obj_id=obj_id, st=spawn_st, end=end)
    #                             stim = dict(zip(return_param, [None]*len(return_param)))
    #                             for param in return_param:
    #                                 if param in stim:
    #                                     stim[param] = tmp[param]
    #                             trial['SpawnedStimuli'].append(stim)


    #             if return_states:
    #                 trial['States'] = []
    #                 for i_id in trl_states:
    #                     trial['States'].append(self.parse_stateid_times(state_id=i_id['id'], st=st, end=end))




    def get_info_per_trial(self, return_eventmarkers=False, choose_trials=False, return_loc = False, start=3000, end=3090, return_spawned=False, spawned_name = 'ImageStimulus', return_param = ['ObjectName', 'Image', 'StimulusCategory'], return_states=False):
        """To select specific trials: choose_trials=['Key']. This will give you all the trials that are True for this key, e.g. 'Correct'. Use '~Key' to select the inverse trials and 'Key=X' to select trials whith specified values, e.g. 'BaseBlock=4'
        To choose alternative start or end of the trial: start=eventmarker, e.g. start=3042"""
        #initiate dictionary
        trial_info=dict()
        
        #extract eventmarkers

        evt, tse, evt_desc, QPC=self.parse_eventmarkers()

        #break eventmarkers into trials----------------------------------------------------------------------------------------
        starttr=3000
        endtr=3090

        starts=np.where(evt==starttr)[0]
        ends=np.where(evt==endtr)[0]
        blocks=np.where((10<=evt)&(evt<50))[0]

        if len(starts)>len(ends):
            starts=starts[0:-1]

        trial_info['StartTs']=tse[starts]
        trial_info['EndTs']=tse[ends]
        tr_starts_ts=trial_info['StartTs']


        ntrials=len(starts)

        evt_tr=np.empty(ntrials, dtype=object)
        evt_ts_tr=np.empty(ntrials, dtype=object)
        evt_desc_tr=np.empty(ntrials, dtype=object)


        ibl=np.argmin(abs(blocks-starts[0]))

        trial_info['BaseBlock']=np.zeros(ntrials,dtype=int)

        for itrl in range(ntrials):
            trial=evt[starts[itrl]:ends[itrl]+1]                
            
            if ibl<len(blocks)-1:

                if blocks[ibl] < starts[itrl] and starts[itrl] < blocks[ibl+1]:
                    block=evt[blocks[ibl]]
                    trial_info['BaseBlock'][itrl]=block-10
                else:
                    block=evt[blocks[ibl+1]]
                    trial_info['BaseBlock'][itrl]=block-10
                    ibl+=1
            else:
                block=evt[blocks[ibl]]
                trial_info['BaseBlock'][itrl]=block-10

                
            evt_tr[itrl]=trial
            evt_ts_tr[itrl]=tse[starts[itrl]:ends[itrl]+1]
            evt_desc_tr[itrl]=evt_desc[starts[itrl]:ends[itrl]+1]

        if return_eventmarkers:
            trial_info['Event'] = evt_tr
            trial_info['EventTime'] = evt_ts_tr
            trial_info['EventDesc'] = evt_desc_tr

        #basic info per trial ---------------------------------------------------------------------------------------------------
  
        for key in (['TrialIndex','Correct','Wrong','Distance','Offset','Left','Right','BaseConditionTarget','BaseConditionDistractor','MorphTarget','MorphDistractor']):

            trial_info[key]=np.zeros(ntrials,dtype=int)
            alt_starts=np.zeros(ntrials,dtype=int)
            alt_starts_ts=np.zeros(ntrials,dtype=float)
            alt_ends=np.zeros(ntrials,dtype=int)
            alt_ends_ts=np.zeros(ntrials,dtype=float)
        

        for itrl,trl in enumerate(evt_tr):
            trial_info['TrialIndex'][itrl]=itrl
            countCond=0
            countMorph=0
            for imark,marker in enumerate(trl):                
                if marker==1:
                    trial_info['Correct'][itrl]=1
                elif marker==2:
                    trial_info['Wrong'][itrl]=1

                elif 50<=marker<100:
                    if countCond == 0:
                        trial_info['BaseConditionTarget'][itrl]=marker-50
                        countCond = 1
                    else:
                        trial_info['BaseConditionDistractor'][itrl]=marker-50

                elif 3400<=marker<=3500:
                    if countMorph == 0:
                        trial_info['MorphTarget'][itrl]=(marker-3400)
                        countMorph=1
                    else:
                        trial_info['MorphDistractor'][itrl]=(marker-3400)

                if start !=3000 or end!=3090:
                    if marker==start:
                        alt_starts[itrl]=1
                        alt_starts_ts[itrl]=evt_ts_tr[itrl][imark]
                    elif marker==end:
                        alt_ends[itrl]=1
                        alt_ends_ts[itrl]=evt_ts_tr[itrl][imark]


        #distances -----------------------------------------------------------------------------------------------------------------

        for itrl, trl in enumerate(evt_tr):
            for imark,marker in enumerate(trl):
                if marker == 3101:
                    trial_info['Distance'][itrl]=trl[imark+2]-10000
                    lr=trl[imark+3]
                    if int(repr(lr)[-1])==4:
                        trial_info['Offset'][itrl]=trl[imark+4]-10000
                    else:
                        trial_info['Offset'][itrl]=-(trl[imark+4]-10000)

        #right/left ---------------------------------------------------------------------------------------------------------------

        for itrl, trl in enumerate (trial_info['Offset']):
            if trl>0:
                trial_info['Left'][itrl]=1   
            elif trl<0:
                trial_info['Right'][itrl]=1
        
        #choose trials------------------------------------------------------------------------------------------------------------

        chosen_trl=np.full(ntrials,True,dtype=bool)

        if choose_trials and type(choose_trials)==list:
            conditions=[]

            for condition in choose_trials:
                if '=' in condition:
                    trials=np.full(ntrials,False,dtype=bool)
                    name=condition.split('=')[0]

                    if '~' in name:
                        name=name.split('~')[-1]

                    value=condition.split('=')[-1]

                    trials[np.where(trial_info[name].astype(str)==value)[0]]=True

                    if '~' in condition:
                        trials=~trials

                elif '~' in condition:
                    name=condition.split('~')[-1]
                    trials=~trial_info[name].astype(bool)

                else:
                    trials=trial_info['{name}'.format(name=condition)].astype(bool)

                conditions.append(trials)
        
            if len(conditions)==1:
                chosen_trl=conditions[0]
                
            elif len(conditions)==2:

                chosen_trl=np.logical_and(conditions[0],conditions[1])

            else:
                chosen_trl=np.logical_and(conditions[0],conditions[1])
                for c in conditions[2:]:
                    chosen_trl=np.logical_and(chosen_trl,c)


        #choose alternative start or end ----------------------------------------------------------------------------------------

        if start !=3000 or end!=3090:
            start_and_end=np.logical_and(alt_starts,alt_ends)
            chosen_trl=np.logical_and(chosen_trl,start_and_end)

            trial_info['StartTs']=alt_starts_ts
            trial_info['EndTs']=alt_ends_ts
        
        for key in trial_info:
            trial_info[key]=trial_info[key][chosen_trl] 
                
        tr_starts_ts=tr_starts_ts[chosen_trl]


        #return paths----------------------------------------------------------------------------------------------------------
        
        if return_loc:

            self.make_id_struct()
            
            for i, name in enumerate (self.all_ids['name']):
                if name.startswith('AnimalCharacter'):
                    AC = name
                    break

            loc, rot, ts =self.parse_position(obj=AC)
            
            trial_info['Location']=np.empty(len(trial_info['StartTs']), dtype=object)
            trial_info['LocationTs']=np.empty(len(trial_info['StartTs']), dtype=object)

            for itrl,trl in enumerate(trial_info['StartTs']):
                startloc=np.where(ts == trl)[0][0]
                starttrloc=np.where(ts==tr_starts_ts[itrl])[0][0]
                endloc=np.where(ts == trial_info['EndTs'][itrl])[0][0]
                trial_info['Location'][itrl]=loc[startloc:endloc,0:2] - loc[starttrloc,0:2]
                trial_info['LocationTs'][itrl]=ts[startloc:endloc]

        # return spawn and states ------------------------------------------------------------------------------------------------------------------------
        
        if return_spawned or return_states:

            all_ends=[]
            for iname in self.all_ids['name']:
                if 'EndTrial' in iname:
                    iname=iname.split('_C_')
                    all_ends.append(iname[0])
            all_ends= list(set(all_ends))
        
            st_ts, st_ind = self.parse_all_state_times(state='StartTrial', times='StateStarted', return_index=True)

            

            end_ind=[]
            for iend in all_ends:

                end_ts, end_i = self.parse_all_state_times(state=iend, times='StateStarted', return_index=True)
                end_ind+=end_i

            #st_ts, st_ind = self.parse_all_state_times(state='StartTrial', times='StateStarted', return_index=True)
            #end_ts, end_ind = self.parse_all_state_times(state='EndTrial', times='StateStarted', return_index=True)

            st_ind=[item for sublist in st_ind for item in sublist]
            end_ind=[item for sublist in end_ind for item in sublist]


            if return_spawned:
                trial_info['SpawnedStimuli'] = np.empty(ntrials,dtype=object)
            if return_states:
                trial_info['States'] = np.empty(ntrials,dtype=object)


            for ist, st in enumerate(st_ind):

                tr_stimuli=[]
                tr_states=[]


                if ist >= len(end_ind):
                    # more start trials than end trials
                    continue
                end = end_ind[ist]

                trl_ids = self.get_ids_per_segment(st=st, end=end)
                trl_states = trl_ids[trl_ids["id_type"] == self.StateName]
                trl_obj = trl_ids[trl_ids["id_type"] == self.ObjectName]


                if return_spawned:
                    
                    for ii, i_id in enumerate(trl_obj):
                        if spawned_name in i_id['name']:
                            obj_id = i_id['id']
                            spawn_st = i_id['start']
                            
                            if spawn_st >= st and spawn_st < end:
                                tmp = self.parse_initial_parameters(obj_id=obj_id, st=spawn_st, end=end)
                                stim = dict(zip(return_param, [None]*len(return_param)))
                                for param in return_param:
                                    if param in stim:
                                        stim[param] = tmp[param]
                                tr_stimuli.append(stim)

                    trial_info['SpawnedStimuli'][ist]=tr_stimuli


                if return_states:
                    for i_id in trl_states:
                        tr_states.append(self.parse_stateid_times(state_id=i_id['id'], st=st, end=end))
                    trial_info['States'][ist]=tr_states



        return trial_info

# %%
