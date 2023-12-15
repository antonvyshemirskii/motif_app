#%%

import os
import numpy as np
import pandas as pd
import copy
import h5py
import shutil
from datetime import datetime as dt

class DTW:
    
    def __init__(self, output_folder='./', gpu_type='p100', zeroing=False, stitched_data=False):
        
        self.__adjust_command = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/g/easybuild/x86_64/Rocky/8/haswell/software/HDF5/1.10.7-gompi-2021a/lib; ' 
        self.output_folder = output_folder
        self.dest_path_out = ''

        try:
            self.working_folder = os.path.abspath(os.path.dirname(__file__))
        except NameError:
            self.working_folder = None

        self.main_executable = 'main'
        self.zeroing = zeroing
        self.stitched_data = stitched_data

        
        # if gpu_type=='p100':
        #     self.main_executable = 'main_p100'
        # else:
        #     self.main_executable = 'main_geforce'
        
    def set_working_folder(self, working_folder):

        self.working_folder = working_folder


    def run_main(self, run_command):
        
        os.system(self.__adjust_command + run_command)

        
    def modif_output(self, file_path, run_type, 
                     dataset_in, num_features, masking_factor):
        
        f = h5py.File(file_path,'a')
        f['run_type'] = run_type
        f['dtw_input'] = dataset_in
        # f['num_features'] = num_features
        # f['masking_factor'] = masking_factor
        f['index_offset'] = int((num_features + 1) / masking_factor)
        f.close()

        
    def run_query(self, query):
        
        #dtw_input =  pd.concat([query, reference], ignore_index=True)
        #file_name = self.make_query_file(query)
        
        num_features = len(query)
        masking_factor = 1
        
        query_file = self.make_query_file(query)
        
        # file_name = os.path.join(self.working_folder, 'dtw_input.csv')
        # dtw_input.to_csv(file_name)
        
        run_command = f'{self.working_folder}/{self.main_executable} {query_file}\
        -masking_factor {masking_factor} -num_features {num_features} -motif_points 1'
        if self.zeroing: run_command = run_command + ' --zeroing'
        if self.stitched_data: run_command = run_command + ' --stitched_data'

        self.run_main(run_command)

        with h5py.File(query_file, 'r') as f:
            masked_distance = np.array(f['masked_distance_vector'])
            input_reference = np.array(f['input_reference'])

            if self.stitched_data:
                stitch_points = np.array(f['stitch_points'], dtype='bool_')
            else:
                stitch_points = None


        masked_distance_trimmed =  masked_distance[int((num_features + 1) - ((num_features + 1) / masking_factor)):]
        #return masked_distance_trimmed

        #masked_distance_trimmed_indexed = pd.Series(index=range(num_features, len(masked_distance_trimmed)+num_features), 
        #                                            data=masked_distance_trimmed)

        masked_distance_trimmed_indexed = pd.Series(index=range(len(masked_distance_trimmed)), 
                                                    data=masked_distance_trimmed)

        return masked_distance_trimmed_indexed, stitch_points

        # index_sequence_
        # path_output = self.save_output()
        # self.modif_output(path_output, 'query', dtw_input, num_features, masking_factor)
        # return path_output

    
    def make_query_file(self, query):
        
        keys, f = self.get_dataset()
        input_reference = self.read_element(f, 'input_reference', 'float32').T

        if self.stitched_data:
            stitch_points = self.read_element(f, 'stitch_points', 'bool_').T

        file_name = os.path.join(self.working_folder, 'dtw_query.h5')
        #query_unsqueeze = query.reshape(-1, 1).astype('float32')
        query_unsqueeze = query.astype('float32')

        print('input_reference shape:', input_reference.shape)
        print('query shape:', query_unsqueeze.astype('float32').shape)

        with h5py.File(file_name, 'w') as f:
            f['input_reference'] = np.concatenate([query_unsqueeze, input_reference], dtype='float32').T
            
            if self.stitched_data:
                f['stitch_points'] = np.concatenate([np.zeros([len(query_unsqueeze)], dtype='bool_'), 
                                                     stitch_points], dtype='bool_').T

        return file_name
    
    
    def run_search(self, reference, num_features, masking_factor):
        
        file_name = os.path.join(self.working_folder, 'dtw_input.csv')
        reference.to_csv(file_name)
        run_command = f'{self.working_folder}/{self.main_executable} {file_name} {masking_factor} {num_features}'
        self.run_main(run_command)
        path_output = self.save_output()
        self.modif_output(path_output, 'search', reference, num_features, masking_factor)
        
        return path_output

        
#     def save_output(self, file_output='dtw_output.h5'):
#         """
#             saves output after main running, 
#             assigns unique name,
#             and returns full path to the output file
        
#             output_folder: full path to the existing folder
#         """
#         current_datatime = dt.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
#         file_output_split = file_output.split('.')
#         id_file_out_name = file_output_split[0] + '_' + current_datatime + '.' + file_output_split[1]
#         full_path_out = os.path.join(self.output_folder, id_file_out_name)
#         self.dest_path_out = shutil.copy(file_output, full_path_out)
        
#         return self.dest_path_out
    

    def load_h5(self, dest_path_out):
    
        self.dest_path_out = dest_path_out
        self.keys, f = self.get_dataset()

        try:
            self.index_sequence = self.read_element(f, 'index_sequence', 'int32')
            self.num_features = self.read_element(f, 'num_features', 'int32')
            self.masking_factor = self.read_element(f, 'masking_factor', 'int32')
        except KeyError:
            self.index_sequence = None
            self.num_features = None
            self.masking_factor = None

        f.close()
        

    def read_element(self, f, key_name, dtype):
        
        element_h5 = f[key_name]
        arr = np.zeros(element_h5.shape, dtype=dtype)
        
        try:
            element_h5.read_direct(arr)
        except OSError:
            if dtype=='object':
                element_h5_list = list(element_h5)
                #element_h5_list = [unicode(x, 'utf-8') for x in element_h5]
                element_h5_list_decoded = [x.decode('utf-8') for x in element_h5_list]
                arr = np.array(element_h5_list_decoded, dtype=dtype)

        return arr
    
    def get_dataset(self):
    
        f = h5py.File(self.dest_path_out,'r')

        return list(f.keys()), f
    
    
    def close_dataset(self, f):
        
        f.close()
        
        
    def purge_files(self, all_files=False):
        
        os.remove(os.path.join(self.working_folder, 'dtw_input.csv'))
        os.remove(os.path.join(self.working_folder, 'dtw_output.h5'))
        
        if all_files: os.remove(self.dest_path_out)
    
    
    def get_rh(self, i):
        
        keys, f = self.get_dataset()

        rh = f['ranked_hits'][f'iter_{i}']
        rh_idx = f['ranked_hits_idx'][f'iter_{i}']
        
        rh_arr = create_scale= np.zeros((rh.shape), dtype='int64')
        rh.read_direct(rh_arr)
        rh_df = pd.DataFrame(rh_arr.T)

        rh_idx_arr = create_scale= np.zeros((rh_idx.shape), dtype='int64')
        rh_idx.read_direct(rh_idx_arr)
        rh_idx_df = pd.DataFrame(rh_idx_arr.T)
        
        
        try:
            run_type = f['run_type'][()].decode('utf-8')
        except KeyError:
            run_type = ''
        
        if run_type == 'query':
            index_offset = f['index_offset'][()]
            rh_idx_df[0] = rh_idx_df[0] - (index_offset - 1)
            
        f.close()
            
        ranked_hits = pd.Series(index=rh_idx_df[0].values, data=rh_df[0].values).sort_values()

        return ranked_hits
    
    
    def get_rh_new(self, i):
        
        keys, f = self.get_dataset()
        rounds = list(f['ranked_hits_idx'].keys())
        shape = f['ranked_hits_idx'][rounds[0]].shape
        
        i_round = int(i / shape[0])
        i_rem = i % shape[0]
        
        rh = f['ranked_hits'][f'round_{i_round}']
        rh_idx = f['ranked_hits_idx'][f'round_{i_round}']
        
        rh_arr = create_scale= np.zeros((rh.shape), dtype='int64')
        rh.read_direct(rh_arr)

        rh_idx_arr = create_scale= np.zeros((rh_idx.shape), dtype='int64')
        rh_idx.read_direct(rh_idx_arr)
        
        try:
            run_type = f['run_type'][()].decode('utf-8')
        except KeyError:
            run_type = ''
        
        if run_type == 'query':
            index_offset = f['index_offset'][()]
            rh_idx_arr = rh_idx_arr - (index_offset - 1)
        
        f.close()
            
        ranked_hits = pd.Series(index=rh_idx_arr[i_rem, :], data=rh_arr[i_rem, :]).sort_values()

        return ranked_hits
        

    def get_scores(self):
        
        keys, f = self.get_dataset()

        shape = f['scores'].shape
        arr = np.zeros(shape, dtype='int64')
        score = f['scores']
        score.read_direct(arr)
        f.close()

        return arr, shape[0]
    
    
    def get_skew(self):
        
        keys, f = self.get_dataset()
        rounds = list(f['skew'].keys())
        base_shape = f['skew'][rounds[0]].shape

        total_len = 0
        
        for round in rounds:
            total_len += f['skew'][round].shape[0]

        #print('total_len', total_len)
            
        arr_master = np.zeros((total_len, ), dtype='float32')
        
        for i, round in enumerate(rounds):
            chunk_shape = f['skew'][round].shape
            arr = np.zeros((chunk_shape[0], ), dtype='float32')
            skew = f['skew'][round]
            skew.read_direct(arr)
            arr_master[i*base_shape[0]:(i*base_shape[0]+chunk_shape[0])] = arr
        
        f.close()

        arr_master_df = pd.Series(index=self.index_sequence[:total_len], data=arr_master)
        
        return arr_master_df
    
    
    def reconstruct_master_df(self):
        
        f = h5py.File(self.dest_path_out, 'r')
            
        grp = f['master_dist']
        
        # index = pd.to_timedelta(self.read_element(grp, 'index', 'int64'))
        index = pd.to_datetime(grp["index"][()].astype("U30"), format='%Y-%m-%dT%H:%M:%S.%f')
        select_files = list(self.read_element(grp, 'select_files', 'object'))
        resample_string = self.read_element(grp, 'resample_string', 'object').reshape(1)[0].decode('utf-8')
        #bodypart = self.read_element(grp, 'bodypart', 'object').reshape(1)[0].decode('utf-8')
        dimensions =  self.read_element(f, 'dimensions', 'int32')

        select_columns = []

        for d in range(dimensions[0]):
            select_columns.append(self.read_element(grp, 'select_column_'+str(d), 'object').reshape(1)[0].decode('utf-8'))

        master_df = pd.DataFrame(index=index) 
        columns = list(self.read_element(grp, 'columns', 'object'))
        
        for col in columns:

            try:
                col_decoded = col.decode('utf-8')
                col_dtype = grp[col].dtype
                master_df[col_decoded] = self.read_element(grp, col_decoded, col_dtype)
            except KeyError:
                pass

        master_df['frame_original'] = self.read_element(grp, 'frame_original', 'int32')
        master_df['filenumber'] = self.read_element(grp, 'filenumber', 'int32')

        f.close()
        
        return master_df, select_files, select_columns, resample_string
    
    
    def get_dtw_params(self):
        
        f = h5py.File(self.dest_path_out, 'r')
        
        num_features = self.read_element(f, 'num_features', 'int32')
        masking_factor = self.read_element(f, 'masking_factor', 'int32')
        
        f.close()
        
        return int(num_features), masking_factor
    

    
