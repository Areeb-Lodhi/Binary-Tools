from .Binary import *

import os
import pandas as pd

from typing import Literal, Union, Tuple
Series = pd.Series

class OConnellUtils:
    @staticmethod 
    def _averagecurve_file(target_obj: 'Binary', filepath: str) -> 'Binary':
        filepath = f'{filepath}/ Average.csv'

        rows = []
        previous_start = 0
        
        
        for quarter in target_obj.quarter_cutoffs.index:
            num = target_obj.quarter_cutoffs['cutoff'][quarter]
            fourier_series = target_obj.fourier_fit(start_index = previous_start, n_points = num - previous_start - 1)
            data = target_obj.get_asym(fourier_series).to_pandas()
            data.index = [f"{target_obj.id.lstrip('0')} Q{quarter}"]
            rows += [data]
            previous_start = num 

        fourier_series = target_obj.fourier_fit()
        data = target_obj.get_asym(fourier_series).to_pandas()
        data.index = [f"{target_obj.id.lstrip('0')} QALL"]
        rows += [data]
        

        datafile = None 
        if os.path.exists(filepath):
            datafile = pd.read_csv(filepath, index_col = 0)
            datafile.index = datafile.index.astype(str)

        if datafile is None:
            data.to_csv(filepath)
            return target_obj

        to_concat = [datafile.drop(index=data.index, errors='ignore')] + rows
        datafile = pd.concat(to_concat, axis = 0) ## if len(to_concat) > 0 else data
        datafile.to_csv(filepath)

        return target_obj 
        
    @staticmethod
    def _singletarget_file(target: Union[int, str], author: Literal['kepler', 'tess', 'k2'], by: Tuple[int, int] = (1, None), filepath: str = f'{os.getcwd()}/INSERT AUTHOR HERE Data', **kwargs) -> 'Binary':
        try:
            target_obj = Binary(target, author, **kwargs) 
        except ValueError as e:
            print(f'Value Error: {e}')
            return None

        n_cyc = by[0]
        n_points = by[1]


        filepath = f'''{filepath.replace('INSERT AUTHOR HERE', author.capitalize())}/{n_cyc} Cycles/{n_points} Points''' if 'Points' not in filepath else filepath
        datafile = None 
        all_cyc_data = []

        errors = [] 

        os.makedirs(filepath, exist_ok=True)

        print(f'Target {target}: {n_cyc} Cycles with {n_points} Points')
        print(f'Status: 0 / {n_cyc}'.ljust(50), end = '\r')
        for cyc in range(0, n_cyc):
            try:
                fourier_series = target_obj.fourier_fit(start_index = cyc * n_points, n_points = n_points)
                data = target_obj.get_asym(fourier_series).to_pandas()
                data['coeffs'] = [fourier_series.coeffs]
                data['quarter'] = [target_obj.active_data['quarter'].iloc[0]]
                data['cyc'] = cyc+1
                all_cyc_data.append(data)
            except BinaryError as e:
                ## print(f'BinaryError: {e}')
                errors.append(cyc)
                continue 
            except TypeError as e:
                ## print(f'TypeError: {e}')
                break

            print(f'Status: {cyc + 1} / {n_cyc}'.ljust(50), end = '\r')

        if datafile is None:
            datafile = pd.concat(all_cyc_data, axis = 0)  
        else:
            datafile = pd.concat(all_cyc_data + [datafile], axis = 0)
        

        print(f'Status: Saving File...'.ljust(50), end = '\r')

        datafile = datafile.set_index('cyc')
        datafile.to_csv(f'{filepath}/{target_obj.id}.csv')
        
        print(f'Status: File Saved'.ljust(50), end = '\r')
        print(f'Status: Finished'.ljust(50), end = '\n')    
        print(f'Errored Cycles: {errors}', end='\n\n')

        if kwargs['get_average']:
            OConnellUtils._averagecurve_file(target_obj, filepath)

        return (target_obj, datafile)
        
    @staticmethod
    def _multitarget_file(targets: Series, author: Literal['kepler', 'tess', 'k2'], by: Tuple[int, int] = (1, None), filepath: str = f'{os.getcwd()}/INSERT AUTHOR HERE Data', **kwargs) -> list['Binary']:
        filepath = f'{filepath.replace("INSERT AUTHOR HERE", author.capitalize())}/{by[0]} Cycles/{by[1]} Points'
        all_target_obj = [] 
            
        for target in targets:
            target_obj = OConnellUtils._singletarget_file(target, author, by, filepath, **kwargs)[0]
            if target_obj is None:
                continue
            target_obj = OConnellUtils._averagecurve_file(target_obj, filepath)
            all_target_obj.append(target_obj)
            
        return all_target_obj

        
    @staticmethod 
    def get_file(target: Union[Series, str, int],  author: Literal['kepler', 'tess', 'k2'], by: Tuple[int, int] = (1, None), filepath: str = f'{os.getcwd()}/INSERT AUTHOR HERE Data', **kwargs) -> Union['Binary', list['Binary']]:
        kwargs.setdefault('get_average', True)
        
        filepath = filepath.replace('INSERT AUTHOR HERE', author.capitalize())
        if isinstance(target, (list, tuple, Series)):
            
            if len(target) > 1: 
                return OConnellUtils._multitarget_file(target, author, by, filepath, **kwargs)
        
            return OConnellUtils._singletarget_file(target[0], author, by, filepath, **kwargs)

        return OConnellUtils._singletarget_file(target, author, by, filepath, **kwargs)

    @staticmethod 
    def plot(file : pd.DataFrame, attributes : Union[Literal['LCA', 'OER', 'ΔI'], list[Literal['LCA', 'OER', 'ΔI']]], name = None, by_quarter = False) -> None:
        attributes = np.atleast_1d(attributes)

        if not by_quarter:
            for attribute in attributes:
                plt.scatter(file[attribute].index, file[attribute])
                plt.title(f'{attribute} vs Cycles {"for " + name if name is not None else ""}')
                plt.show()

            return

        colors = ["red","blue","green","orange","purple","brown","pink","gray","olive","cyan","magenta","gold","navy","lime","indigo","teal","maroon","turquoise","darkorange","darkgreen","slateblue"]
        plts = []
        for attribute in attributes:
            plt.figure()
            for q in range(0, 21):
                if q not in file['quarter'].values:
                    continue 
    
                plot_data = file[file['quarter'] == q]
                plt.scatter(plot_data[attribute].index, plot_data[attribute], label=f'Quarter {q}', color=colors[q])
                plt.title(f'{attribute} vs Cycles {"for " + name if name is not None else ""}')

            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.xlabel("Cycles")
            plt.ylabel(attribute)
            plts.append(plt)

        return plts

    def plot_fourier_coeffs(file : pd.DataFrame, an = [], bn = [], name = None, by_quarter = False) -> None:
        def func_to_vectorize(x):
            return FourierSeries.construct_from_str(x)

        vectorized_func = np.vectorize(func_to_vectorize)
            
        if not by_quarter:
            coeff_data = vectorized_func(plot_data['coeffs'].to_numpy())
            plts = []
            for term in an:
                fig = plt.figure()
                coeff_data_temp = coeff_data[:, 2*term -1 if term != 0 else 0]
                plt.scatter(file['coeffs'].index, coeff_data_temp)
                plt.title(f'a{term} vs Cycles {"for " + name if name is not None else ""}')
                plts.append(fig)

            for term in bn:
                fig = plt.figure()
                coeff_data_temp = coeff_data[:, 2*term]
                plt.scatter(file['coeffs'].index, coeff_data_temp)
                plt.title(f'b{term} vs Cycles {"for " + name if name is not None else ""}')
                plts.append(fig)

            return plts

        colors = ["red","blue","green","orange","purple","brown","pink","gray","olive","cyan","magenta","gold","navy","lime","indigo","teal","maroon","turquoise","darkorange","darkgreen","slateblue"]
        plts = []
        for term in an:
            fig = plt.figure()
            for q in range(0, 21):
                if q not in file['quarter'].values:
                    continue 
    
                plot_data = file[file['quarter'] == q]
                coeff_data = vectorized_func(plot_data['coeffs'].to_numpy())
                coeff_data = coeff_data[:, 2*term - 1 if term != 0 else 0]
                
                plt.scatter(plot_data['coeffs'].index, coeff_data, label=f'Quarter {q}', color=colors[q])
                plt.title(f'a{term} vs Cycles {"for " + name if name is not None else ""}')

            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.xlabel("Cycles")
            plt.ylabel(f'a{term}')
            plts.append(fig)

        for term in bn:
            fig = plt.figure()
            for q in range(0, 21):
                if q not in file['quarter'].values:
                    continue 
    
                plot_data = file[file['quarter'] == q]
                coeff_data = vectorized_func(plot_data['coeffs'].to_numpy())
                coeff_data = coeff_data[:, 2*term]
                
                plt.scatter(plot_data['coeffs'].index, coeff_data, label=f'Quarter {q}', color=colors[q])
                plt.title(f'b{term} vs Cycles {"for " + name if name is not None else ""}')

            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.xlabel("Cycles")
            plt.ylabel(f'b{term}')
            plts.append(fig)

        return plts
            

