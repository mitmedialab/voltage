import time
import math
import pandas as pd
import tifffile as tiff


class Timer:

    def __init__(self, time_filename=None):
        self.time_filename = time_filename
        if(time_filename is None):
            self.df = None
        else:
            if(time_filename.exists()):
                self.df = pd.read_csv(time_filename)
            else:
                self.df = pd.DataFrame()
        self.stage2time = {}


    def start(self):
        self.tic = time.perf_counter()


    def stop(self, stage_name):
        toc = time.perf_counter()
        elapsed_time = toc - self.tic
        self.stage2time[stage_name] = elapsed_time
        print('Stage %s done: %.2f seconds' % (stage_name, elapsed_time))
        if(self.df is not None):
            self.df[stage_name] = [elapsed_time]
            self.df.to_csv(self.time_filename, index=False)


    def skip(self, stage_name):
        msg = 'Stage %s skipped' % stage_name
        if(self.df is not None and stage_name in self.df.columns):
            elapsed_time = self.df.at[0, stage_name]
            self.stage2time[stage_name] = elapsed_time
            msg += ' (previously took %.2f seconds)' % elapsed_time
        print(msg)


    def save(self, data_filename):
        if(self.df is None):
            return
        elapsed_time_total = sum(self.stage2time.values())
        print('Total: %.2f seconds' % elapsed_time_total)
        if('Dataset ID' not in self.df.columns):
            self.df.insert(0, 'Dataset ID', data_filename.stem)
        self.df['Total [sec]'] = elapsed_time_total
        with tiff.TiffFile(data_filename) as tif:
            num_frames = len(tif.pages)
        self.df['# frames'] = num_frames
        self.df['Time/frame [msec]'] = elapsed_time_total * 1000 / num_frames
        if(elapsed_time_total == 0):
            self.df['Frame rate [fps]'] = math.nan
        else:
            self.df['Frame rate [fps]'] = num_frames / elapsed_time_total
        data_size_in_gb = data_filename.stat().st_size / (1024 ** 3)
        self.df['Data size [GB]'] = data_size_in_gb
        self.df['Time/size [sec/GB]'] = elapsed_time_total / data_size_in_gb
        self.df.to_csv(self.time_filename, index=False)
