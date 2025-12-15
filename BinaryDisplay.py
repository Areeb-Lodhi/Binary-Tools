import numpy as np
import pandas as pd
import io 
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from Binary import *
from Oconnell import *


class _Meta(type):
    def __str__(cls):
        cls.display()
        return ''

class BinaryDisplay(metaclass=_Meta): 
    entries = [] 
    author = None
    
    file = None 
    file_col = None 
    
    basic = ['Animate', 'Light Curve']
    advanced = ['Asymmetry', 'Annotations', 'Gaia Data', "O'Connell Data"]
    
    processes = {
        'Animate' : {
            'Asymmetry' : False,
            'Light Curve' : False,
            'Annotations' : {
                'Gaia Data' : False,
                'Asymmetry Data' : {
                    'LCA' : False,
                    'OER' : False,
                    'ΔI' : False,
                    'Assosiated Errors' : False,
                    'Value Format' : 'Raw'
                }
            }
        }, 
        "O'Connell Data" : False, 
        'Fourier Coefficients' : False
    } 

    cuts = {}

    _main_output = widgets.Output()

    @staticmethod
    def _proccess():
        if BinaryDisplay.file is not None:
            content = BinaryDisplay.file.value[0]['content']
            file = pd.read_csv(io.BytesIO(content))
            file_column = BinaryDisplay.file_col.value
            
            for col in [file_column] + list(BinaryDisplay.cuts.keys()):
                if col not in file.columns:
                    print(f'An invalid column was input for this file')
                    BinaryDisplay._main_output.clear_output()
                    BinaryDisplay.display()
                    return 

            mask = file.index >= -1
            for col, cut in BinaryDisplay.cuts.items():
                file_at_col = file[col].apply(float)
                if cut[0] != 0:
                    mask = mask & (file_at_col >= cut[0])
                if cut[1] != 0:
                    mask = mask & (file_at_col <= cut[1])
                    
            
            BinaryDisplay.entries = file[mask][BinaryDisplay.file_col.value.strip()].to_list()

            if len(BinaryDisplay.entries) == 0:
                print('No targets found within the provided parameters')
                BinaryDisplay._main_output.clear_output()
                BinaryDisplay.display()
                return 

        home = widgets.Button(
            description='',
            disabled=False,
            button_style='',
            icon='home',
            layout=widgets.Layout(width='32px', height='32px', padding='0px', margin='0px')
        )

        cycles_int = widgets.BoundedIntText(
            value=10,
            min=0,
            max=1e10,
            disabled=False,
            layout=widgets.Layout(width='30%', margin='0px', padding='0px'),
            style={'text_align': 'center'}
        )
        
        points_int = widgets.BoundedIntText(
            value=100,
            min=0,
            max=1e10,
            description='Cycles x ',
            disabled=False,
            layout=widgets.Layout(width='40%', margin='0px', padding='0px'),
            style={'description_width' : 'initial', 'text_align' : 'center'}
        )
        
        points_label = widgets.Label(
            'Points',
            layout=widgets.Layout(width='auto', margin='0px', padding='0px')
        )
        
        harm_int = widgets.BoundedIntText(
            value=10,
            min=0,
            max=1e10,
            disabled=False,
            layout=widgets.Layout(width='70%', margin='auto 0px 0px 0px', padding='0px'),
            style={'text_align': 'center'}
        )
        
        harm_label = widgets.Label(
            'Harmonics',
            layout=widgets.Layout(width='20%', margin='0px', padding='0px')
        )
        
        submit = widgets.Button(
            description='Submit',
            disabled=False,
            button_style='success',
            icon='check',
            layout=widgets.Layout(width='100%', margin='0px')
        )

        info_group = widgets.VBox(
            [
                home,
                widgets.VBox(
                    [
                        widgets.HBox([cycles_int, points_int, points_label],
                                     layout=widgets.Layout(justify_content='flex-start', margin='0px', padding='0px')),
                        widgets.HBox([harm_int, harm_label],
                                     layout=widgets.Layout(justify_content='flex-start', margin='0px', padding='0px'))
                    ],
                    layout=widgets.Layout(
                        border='2px solid gray',
                        padding='10px',
                        margin='5px 0px 5px 0px',
                        width='100%'
                    )
                ),
                widgets.HBox([submit], layout=widgets.Layout(justify_content='center'))
            ],
            layout=widgets.Layout(justify_content='center', width='100%', padding='10px')
        )
        
        CURRENT_WINDOW = widgets.VBox(
            [widgets.HBox([info_group], layout=widgets.Layout(justify_content='center', width='100%'))],
            layout=widgets.Layout(width='500px')
        )

        def get_asymmetry_selection_bar(file : pd.DataFrame, target): 
            plot_lca = widgets.Button(
                description='LCA',
                disabled=False,
                button_style='success',
                icon='',
                layout=widgets.Layout(width='100%', margin='2.5px')
            )
            plot_oer = widgets.Button(
                description='OER',
                disabled=False,
                button_style='success',
                icon='',
                layout=widgets.Layout(width='100%', margin='2.5px')
            )
            plot_I = widgets.Button(
                description='ΔI',
                disabled=False,
                button_style='success',
                icon='',
                layout=widgets.Layout(width='100%', margin='2.5px')
            )
            plot_coeffs = widgets.Button(
                description='Fourier Coefficients',
                disabled=False,
                button_style='success',
                icon='',
                layout=widgets.Layout(width='100%', margin='2.5px')
            )

            home_selection = [plot_lca, plot_oer, plot_I, plot_coeffs]

            selection_bar = widgets.HBox(
                home_selection,
                layout = widgets.Layout( border = '2px solid gray', margin = '2.5px')
            )

            plots = widgets.VBox([], layout=widgets.Layout(border=None, margin='2.5px', align_items = 'center', justify_content = 'center'))

            plots_widgets = widgets.Output()

            def plot_vs_cycles(b):
                
                if b != plot_coeffs:
                    plots.layout.border = '2px solid gray'
                    plt = OConnellUtils.plot(file, attributes=[b.description], name=target, by_quarter=True)[0]
                    with plots_widgets:
                        display(plt.gcf())
                    plots.children = [plots_widgets]
                    b.disabled = True

                else:
                    coeff_selection = []
                    for char in ['a', 'b']:
                         coeff_selection.append(
                                widgets.Text(
                                value='',
                                placeholder='Use , to seperate and - for range',
                                description=f'{char}n',
                                disabled=False,
                                layout=widgets.Layout(width='50%'),
                                style={'description_width' : 'initial'}
                            )
                         )

                    coeff_selection.append(
                        widgets.Button(
                            description='Plot Coeffs',
                            disabled=False,
                            button_style='success',
                            icon='',
                            layout=widgets.Layout(width='100%', margin='2.5px')
                        )
                    )

                    selection_bar.children = coeff_selection

                    def submit(_):
                        an_terms = np.atleast_1d(coeff_selection[0].value.strip().split())
                        bn_terms = np.atleast_1d(coeff_selection[1].value.strip().split())

                        try:
                            for index, term in enumerate(an_terms):
                                if '-' in term:
                                    temp = term.split('-')
                                    an_terms[index:index] = range(int(temp[0].strip()), int(temp[1].strip())+1)
                                    continue 
    
                                an_terms[index] = int(an_terms[index].strip())
    
                            for index, term in enumerate(bn_terms):
                                if '-' in term:
                                    temp = term.split('-')
                                    bn_terms[index:index] = range(int(temp[0].strip()), int(temp[1].strip())+1)
                                    continue 
    
                                bn_terms[index] = int(bn_terms[index].strip())
                    
                            plts = OConnellUtils.plot_fourier_coeffs(file, an = an_terms, bn = bn_terms, name = target, by_quarter = True)
                            plots.layout.border = '2px solid gray'
                            for plt in plts:
                                with plots_widgets:
                                    display(plt.gcf())
                                plots.children = [plots_widgets]
                            b.disabled = True
                            selection_bar.children = home_selection
                        except Exception as e:
                            print(e)
                            coeff_selection[0].value = ''
                            coeff_selection[1].value = ''

                    coeff_selection[2].on_click(submit)

            plot_lca.on_click(plot_vs_cycles)
            plot_oer.on_click(plot_vs_cycles)
            plot_I.on_click(plot_vs_cycles)
            plot_coeffs.on_click(plot_vs_cycles)

            asym_selection_display = widgets.VBox( [selection_bar, plots], layout = widgets.Layout(border=None, margins='25px') )
            return asym_selection_display


        def check_submit(_):
            nonlocal CURRENT_WINDOW
            
            n_cyc = cycles_int.value
            n_points = points_int.value 
            harm = harm_int.value 
            
            if n_cyc > 0 and n_points > 0 and harm > 0:

                cycles_int.disabled = True
                points_int .disabled = True
                harm_int.disabled = True

                info_group.children = info_group.children[:-1]

                check_oconnell =  BinaryDisplay.processes["O'Connell Data"]
                check_animate = True in list(BinaryDisplay.processes['Animate'].values())
                check_fourier = BinaryDisplay.processes['Fourier Coefficients']

                animations_selected = BinaryDisplay.processes['Animate']
                annotations_selected =  BinaryDisplay.processes['Animate']['Annotations'] 
                asymmetry_selected = BinaryDisplay.processes['Animate']['Annotations']['Asymmetry Data']

                for target in BinaryDisplay.entries:

                    if check_oconnell:
                        target_obj, file = OConnellUtils.get_file(target, BinaryDisplay.author, by=(n_cyc,n_points), get_average=True)
                        asymmetry_selection_bar = get_asymmetry_selection_bar(file, target)
                        
                    if check_animate:
                        asymmetry_selection_bar = None 
                        if check_oconnell: 
                            ani, btn = AsymmetryData.animate_file(file, target_obj, by=(n_cyc,n_points), **animations_selected, **annotations_selected, **asymmetry_selected)
                            asymmetry_selection_bar = get_asymmetry_selection_bar(file, target)
                            

                        else:  
                            target_obj = Binary(target, BinaryDisplay.author, harm=harm)
                            ani, btn = AsymmetryData.animate_nonfile(target_obj, by=(n_cyc,n_points), **animations_selected, **annotations_selected, **asymmetry_selected)
                        
                        with BinaryDisplay._main_output:
                            display(btn)
                            display(HTML(ani.to_jshtml()))
            
                    elif check_fourier:
                        pass
                    
                    with BinaryDisplay._main_output:
                        display(asymmetry_selection_bar)

        def exit(_):
            BinaryDisplay._main_output.clear_output()
            BinaryDisplay.display()
            
        submit.on_click(check_submit)
        home.on_click(exit)
        
        with BinaryDisplay._main_output:
            display(CURRENT_WINDOW)

                
    
    @staticmethod 
    def _target_selection_window():
        tab_titles = np.array(['Manual Input', 'CSV File'])
        
        list_of_criterion = {}
        possible_cuts = ['Period', 'Morph']
        
        # Items in the target selection 
        
          # Manual Input Tab
        manual_target_output = widgets.Label(value='Entered Targets: ')
        
        manual_target_input = widgets.Text(
            value='',
            placeholder='Ex: 11560447, Wasp-12b, etc.',
            description='Target Name:',
            disabled=False,
            layout=widgets.Layout(width='50%'),
            style={'description_width' : 'initial'}
        )
        
        manual_target_submit_button = widgets.Button(
            description='Submit',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        
          # CSV Input Tab
        csv_target_upload_label = widgets.Label('Target CSV File:')
        
        csv_target_upload = widgets.FileUpload(
            accept='.csv', 
            multiple=False 
        )
        
        BinaryDisplay.file_col = csv_target_column_query = widgets.Text(
            value='',
            placeholder='Ex: KIC, TIC, etc.',
            description='Column Name:', 
            disabled=False,
            style={'description_width' : 'initial'}
        )

        
        # Items in Author Selection 
        
        author_selection_dropdown = widgets.Dropdown(
            options=[('Kepler', 'kepler'), ('Tess', 'tess'), ('K2', 'k2')],
            value=None,
            description='Mission:',
            style={'description_width' : 'initial'}
        )
        
        # Items in Cut Selection 
        
        add_cut_dropdown = widgets.Dropdown(
            options=[('Period', 'period'), ('Morph', 'morph')],
            value=None,
            description='Add Cut:',
            style={'description_width' : 'initial'}
        )
        
        add_cut_submit = widgets.Button(
            description='Submit',
            disabled=True,
            button_style='success', 
            tooltip='Click me',
            icon='check'
        )
        
        # Group items in Target Selection 
        
          # Manual Target Grouping 
        manual_target_box = widgets.HBox([manual_target_input, manual_target_submit_button])
        manual_target_box = widgets.VBox([manual_target_box, manual_target_output])
        
          # CSV Target Grouping 
        csv_target_box =  widgets.HBox([csv_target_upload_label, csv_target_upload])
        csv_target_box = widgets.VBox([csv_target_box, csv_target_column_query])
        
        # Group Items in Cut Selection 
        
        cuts_box = widgets.VBox([])
        cut_window = widgets.VBox([widgets.HBox([add_cut_dropdown, add_cut_submit]), cuts_box])
        
        # Tab Creation for Target Selection 
        tab_contents = [manual_target_box, csv_target_box]
        
        target_tab = widgets.Tab()
        target_tab.children = tab_contents
        target_tab.titles = list(tab_titles)

        # Group All Items 

        window_group = widgets.VBox(
            [
                target_tab, 
                author_selection_dropdown
            ],
            layout=widgets.Layout(
                border='2px solid gray',
                padding='10px',
                margin='0px',
                width='600px'
            )
        )
        
        # Click/Observation Function for Target Selection 
        
        def del_tab(titles):
            nonlocal tab_titles
            titles = np.atleast_1d(titles)
        
            children = list(target_tab.children)
        
            new_children = []
            new_titles = []
            for i, child in enumerate(children):
                if tab_titles[i] not in titles:
                    new_children.append(child)
                    new_titles.append(tab_titles[i])
            
            target_tab.children = new_children
        
            for i, t in enumerate(new_titles):
                target_tab.set_title(i, t)
        
            tab_titles = new_titles
            
        
        def append_value(_):
            nonlocal tab_titles
            
            value = manual_target_input.value.strip()
            
            if len(BinaryDisplay.entries) == 0 and value:
                del_tab(tab_titles[tab_titles != 'Manual Input'])
                add_cut_submit.disabled = False
                
            if value and value not in BinaryDisplay.entries:
                BinaryDisplay.entries.append(value)
                manual_target_output.value = f'Entered Targets: {", ".join(BinaryDisplay.entries)}'

            manual_target_input.value = ''
            
        
        def uploaded_file(change):
            nonlocal tab_titles
            nonlocal add_cut_submit

            add_cut_submit.disabled = False            
            
            if len(tab_titles) > 1:
                window_group.children += (cut_window,)
                del_tab(tab_titles[tab_titles != 'CSV File'])
                BinaryDisplay.file = change['owner']


        #  Click/Observation Function for Author Selection 
        def update_author(change):
            BinaryDisplay.author = change['owner'].value
        
        # Click/Observation Function for Cut Selection 
        
        def add_cut(change=None):
            value = add_cut_dropdown.value
            if not value or value in list_of_criterion.keys(): 
                return 
        
            list_of_criterion[value] = []
        
            # Create items for new cut 
        
            new_box_label = widgets.Label(f'{value.capitalize()} Cut:')
        
            new_box_criterion_value1 = widgets.BoundedFloatText(
                value=None,
                min=0.0,
                max=1e10,
                description='Min:',
                disabled=False
            )
            
            new_box_criterion_value2 = widgets.BoundedFloatText(
                value=None,
                min=0.0,
                max=1e10,
                description='Max:',
                disabled=False
            )   
        
            new_box_criterion_unit1 = widgets.Dropdown(
                options=[('years', 'years'), ('days', 'days'), ('hours', 'hours')],
                value='days',
                layout=widgets.Layout(width='auto'),
            )

            new_box_criterion_col = widgets.Text(
                value='',
                placeholder='Period, Morph, p, morph, etc.',
                description='Column:',
                disabled=False
            )
        
            new_box_criterion_submit =  widgets.Button(
                description='Submit Criteria',
                disabled=True,
                button_style='success', 
                tooltip=f'Submit Criteria for {value}',
                layout=widgets.Layout(margin='0px 0px 0px 90px', width='134px'),
                icon='check'
            )
        
            new_box_criterion_remove =  widgets.Button(
                description='Remove Criteria',
                disabled=False,
                button_style='danger', 
                tooltip=f'Remove Criteria for {value}',
                layout=widgets.Layout(margin='0px 0px 0px 2px', width='134px'),
                icon='times'
            )
        
            # Group New Cut 
        
            new_box_criterion1 = widgets.HBox([new_box_criterion_value1] + ([new_box_criterion_unit1] if value != 'morph' else []))
            new_box_criterion2 = widgets.HBox([new_box_criterion_value2] + ([new_box_criterion_unit1] if value != 'morph' else []))
            new_box_criterion_buttons = widgets.HBox([new_box_criterion_submit, new_box_criterion_remove])
        
            new_box = widgets.VBox([new_box_label, new_box_criterion1, new_box_criterion2, new_box_criterion_col, new_box_criterion_buttons])
            cuts_box.children += (new_box,)
        
            # Observe/Click Function for New Cut 
        
            def check_dropdown(_):
                if new_box_criterion_value1.value == new_box_criterion_value2.value == 0 or new_box_criterion_col.value=='':
                    new_box_criterion_submit.disabled = True
                else:
                    new_box_criterion_submit.disabled = False 
        
            def remove_criterion(_):
                cuts_box.children = tuple(child for child in cuts_box.children if child != new_box)
                del list_of_criterion[value]
        
            def submit_criterion(_):
                factor = 365.25 if new_box_criterion_unit1.value == 'years' else 1/24 if new_box_criterion_unit1.value == 'hours' else 1
                BinaryDisplay.cuts[new_box_criterion_col.value] = [new_box_criterion_value1.value * factor, new_box_criterion_value2.value * factor]
                
                new_box_criterion_value1.disabled = True
                new_box_criterion_value2.disabled = True
                new_box_criterion_unit1.disabled = True
                new_box_criterion_col.disabled = True
                new_box_criterion_submit.disabled = True

        
            # Observe/Click Establishment Statements 
            
            new_box_criterion_value1.observe(check_dropdown, names='value')
            new_box_criterion_value2.observe(check_dropdown, names='value')
            new_box_criterion_col.observe(check_dropdown, names='value')
            new_box_criterion_remove.on_click(remove_criterion)
            new_box_criterion_submit.on_click(submit_criterion)
        
        # Click/Observe Assignments for Target Selection 
        
          # Manual Target Selection Assignments 
        manual_target_input.continuous_update = False
        manual_target_input.observe(append_value, names='value')
        manual_target_submit_button.on_click(append_value)
        
          # CSV Target Selection Assignments 
        csv_target_upload.observe(uploaded_file, names='value')

        # Click/Observe Assignments for Author Selection

        author_selection_dropdown.observe(update_author, names='value')
        
        # Click/Observe Assignments for Cut Selection 
        
        add_cut_submit.on_click(add_cut)

        return window_group

    @staticmethod
    def _proccess_selection_window():
        basic_checkbox = widgets.Checkbox(
                    value=False,
                    description='Basic Analysis',
                    tooltip='Auto selects processes needed for a basic analysis',
                    disabled=False,
                    indent=False,
                    layout=widgets.Layout(width='auto')
                )
        advanced_checkbox = widgets.Checkbox(
                    value=False,
                    description='Advanced Analysis',
                    tooltip='Auto selects processes needed for an advanced analysis',
                    disabled=False,
                    indent=False,
                    layout=widgets.Layout(width='auto')
                )   
        
        proccess_checkboxes = []
        for proccess in BinaryDisplay.processes.keys():
            proccess_checkboxes.append(
                widgets.Checkbox(
                    value=False,
                    description=proccess,
                    disabled=False,
                    indent=False,
                    layout=widgets.Layout(width='30ch')
                )
            )
        
        animate_checkboxes = [] 
        for proccess in BinaryDisplay.processes['Animate'].keys():
            animate_checkboxes.append(
                widgets.Checkbox(
                    value=False,
                    description=proccess,
                    disabled=False,
                    indent=False,
                    layout=widgets.Layout(margin='0px 0px 0px 40px', width='20ch')
                )
            )
        
        annotations_checkboxes = [] 
        for proccess in BinaryDisplay.processes['Animate']['Annotations'].keys():
            annotations_checkboxes.append(
                widgets.Checkbox(
                    value=False,
                    description=proccess,
                    disabled=False,
                    indent=False,
                    layout=widgets.Layout(margin='0px 0px 0px 80px', width='20ch')
                )
            )

        asymmetry_checkboxes = []
        for proccess in list(BinaryDisplay.processes['Animate']['Annotations']['Asymmetry Data'].keys())[:-1]:
            asymmetry_checkboxes.append(
                widgets.Checkbox(
                    value=False,
                    description=proccess,
                    disabled=False,
                    indent=False,
                    layout=widgets.Layout(margin='0px 0px 0px 120px', width='20ch')
                )
            )

        asymmetry_checkboxes.append( 
            widgets.Dropdown(
                options=[('Raw', 'raw'), ('% from Average', '%')],
                value='raw',
                description='Value Format: ',
                style={'description_width' : 'initial'},
                layout=widgets.Layout(margin='0px 0px 0px 120px', width='20ch')
            ) 
        )
            
            
        animate_group = widgets.VBox([proccess_checkboxes[0]])
        
        def add_next_option(change):
            nonlocal proccess_checkboxes
            nonlocal animate_checkboxes
            nonlocal annotations_checkboxes
            nonlocal animate_group 
            
            if change['new']:

                if annotations_checkboxes[-1].value:
                    animate_group.children += (*asymmetry_checkboxes,)
                
                elif animate_checkboxes[-1].value:
                    animate_group.children += (*annotations_checkboxes,)
                    BinaryDisplay.processes['Animate']['Annotations'] = {
                        'Gaia Data' :  annotations_checkboxes[0].value,
                        'Asymmetry Data' : {}
                    }
                    
                elif proccess_checkboxes[0].value:
                    animate_group.children += (*animate_checkboxes,)
                    BinaryDisplay.processes['Animate'] = {
                        'Asymmetry' : animate_checkboxes[0].value,
                        'Light Curve' : animate_checkboxes[1].value,
                        'Annotations' : {}
                    }

                BinaryDisplay.processes['Animate']['Annotations']['Asymmetry Data'] = {
                    'LCA' : asymmetry_checkboxes[0].value,
                    'OER' : asymmetry_checkboxes[1].value,
                    'ΔI' : asymmetry_checkboxes[2].value,
                    'Assosiated Errors' : asymmetry_checkboxes[3].value,
                    'Value Format' : asymmetry_checkboxes[4].value
                }
            
        def update_status(change):
            BinaryDisplay.processes = {
                'Animate' : {
                    'Asymmetry' : animate_checkboxes[0].value,
                    'Light Curve' : animate_checkboxes[1].value,
                    'Annotations' : {
                        'Gaia Data' : annotations_checkboxes[0].value,
                        'Asymmetry Data' : {
                            'LCA' : asymmetry_checkboxes[0].value,
                            'OER' : asymmetry_checkboxes[1].value,
                            'ΔI' : asymmetry_checkboxes[2].value,
                            'Assosiated Errors' : asymmetry_checkboxes[3].value,
                            'Value Format' : asymmetry_checkboxes[4].value
                        }
                    }
                }, 
                "O'Connell Data" : proccess_checkboxes[1].value, 
                'Fourier Coefficients' : proccess_checkboxes[2].value
            } 
        
        def remove_option(change):
            nonlocal proccess_checkboxes
            nonlocal animate_checkboxes
            nonlocal annotations_checkboxes
            nonlocal animate_group 
            
            if not change['new']:
                
                if not proccess_checkboxes[0].value: 
                    animate_group.children = [proccess_checkboxes[0]]
                    BinaryDisplay.processes['Animate'] = {
                        'Asymmetry' : False,
                        'Light Curve' : False,
                        'Annotations' : {
                            'Gaia Data' : False,
                            'Asymmetry Data' : {}
                        }
                    }
                    for selection in animate_checkboxes:
                        selection.value = False
                        
                elif not animate_checkboxes[-1].value:
                    animate_group.children = [proccess_checkboxes[0], *animate_checkboxes]
                    BinaryDisplay.processes['Animate']['Annotations'] = {
                        'Gaia Data' : False,
                        'Asymmetry Data' : {}
                    }
                    for selection in annotations_checkboxes:
                        selection.value = False

                elif not annotations_checkboxes[-1].value:
                    animate_group.children = [proccess_checkboxes[0], *animate_checkboxes, *annotations_checkboxes]

                BinaryDisplay.processes['Animate']['Annotations']['Asymmetry Data'] = {
                    'LCA' : False,
                    'OER' : False,
                    'ΔI' : False,
                    'Assosiated Errors' : False,
                    'Value Format' : 'Raw'
                }
                    
                for selection in asymmetry_checkboxes[:-1]:
                    selection.value = False
        def analysis_type(change): 
            nonlocal proccess_checkboxes
            nonlocal animate_checkboxes
            nonlocal annotations_checkboxes
            nonlocal advanced_checkbox
            nonlocal basic_checkbox
            
            if change['new']:
                proccess_checkboxes[2].value = False

                BinaryDisplay.processes['Animate']['Annotations']['Asymmetry Data'] = {
                    'LCA' : False,
                    'OER' : False,
                    'ΔI' : False,
                    'Assosiated Errors' : False,
                    'Value Format' : 'Raw'
                }
    
                if advanced_checkbox == change['owner']:
                    basic_checkbox.disabled = True 

                    proccess_checkboxes[0].value = True
                    proccess_checkboxes[1].value = True
                    
                    animate_checkboxes[0].value = True
                    animate_checkboxes[1].value = True
                    animate_checkboxes[2].value = True

                    annotations_checkboxes[0].value = True

                else:
                    advanced_checkbox.disabled = True

                    proccess_checkboxes[0].value = True
                    animate_checkboxes[1].value = True

                    proccess_checkboxes[1].value = False
                    
                    animate_checkboxes[0].value = False
                    animate_checkboxes[2].value = False

                    annotations_checkboxes[0].value = False

            else:
                advanced_checkbox.disabled = False
                basic_checkbox.disabled = False

        basic_checkbox.observe(analysis_type, names='value')
        advanced_checkbox.observe(analysis_type, names='value')
        
        proccess_checkboxes[0].observe(add_next_option, names='value')
        proccess_checkboxes[0].observe(remove_option, names='value')
        
        animate_checkboxes[-1].observe(add_next_option, names='value')
        animate_checkboxes[-1].observe(remove_option, names='value')

        annotations_checkboxes[-1].observe(add_next_option, names='value')
        annotations_checkboxes[-1].observe(remove_option, names='value')

        for checkbox in proccess_checkboxes + animate_checkboxes + annotations_checkboxes + asymmetry_checkboxes:
            checkbox.observe(update_status, names='value')
        analysis_selection_group = widgets.VBox(
            [ 
                basic_checkbox, 
                advanced_checkbox
            ],
            layout=widgets.Layout(
                border='2px solid gray',
                padding='10px',
                margin='2.5px 0px 2.5px 0px',
                width='25ch'
            )
        )
        proccesses_selection_group = widgets.VBox(
            [ 
                animate_group,  
                *proccess_checkboxes[1:]
            ],
            layout=widgets.Layout(
                border='2px solid gray',
                padding='10px',
                margin='2.5px 0px 5px 0px',
                width='400px'
            )
        )
        
        window_group = widgets.VBox(
            [ 
                analysis_selection_group,
                proccesses_selection_group
            ]
        )
        return window_group

    @staticmethod 
    def display():
        BinaryDisplay.entries = []
        BinaryDisplay.processes = {
            'Animate' : {
                'Asymmetry' : False,
                'Light Curve' : False,
                'Annotations' : {
                    'Gaia Data' : False,
                    'Asymmetry Data' : {
                        'LCA' : False,
                        'OER' : False,
                        'ΔI' : False,
                        'Assosiated Errors' : False,
                        'Value Format' : 'Raw'
                    }
                }
            }, 
            "O'Connell Data" : False, 
            'Fourier Coefficients' : False
        } 
        BinaryDisplay.file = None
            
        CURRENT_WINDOW_INDEX = 0
        windows = [BinaryDisplay._target_selection_window(), BinaryDisplay._proccess_selection_window()]

        refresh = widgets.Button(
            description='',
            button_style='',
            tooltip='Next Page',
            icon='refresh',
            layout=widgets.Layout(width='50px', height='32px', margin='0px', padding='0px')
        )

        next_pg = widgets.Button(
            description='',
            button_style='',
            tooltip='Next Page',
            icon='arrow-right',
            layout=widgets.Layout(width='50px', height='32px', margin='0px', padding='0px')
        )

        prev_pg = widgets.Button(
            description='',
            button_style='',
            tooltip='Prev Page',
            icon='arrow-left',
            layout=widgets.Layout(width='50px', height='32px', margin='0px', padding='0px')
        )

        submit = widgets.Button(
            description='Submit',
            button_style='success',
            tooltip='Start Analysis',
            icon='',
            layout=widgets.Layout(justify_content='center', width='auto', margin='0px', padding='0px')
        )
        
        next_pg_formatted = widgets.HBox([next_pg], layout=widgets.Layout(width='auto', justify_content='flex-end'))
        prev_pg_formatted = widgets.HBox([prev_pg], layout=widgets.Layout(width='auto', justify_content='flex-start'))
        
        pg_group = widgets.HBox([prev_pg, next_pg], layout=widgets.Layout(width='auto', justify_content='space-between'))

        windows_with_page_buttons = []
        for i, window in enumerate(windows):
            window_width = window.layout.width or '600px'
            if i == 0: 
                windows_with_page_buttons.append( widgets.VBox([refresh, window, next_pg_formatted], layout=widgets.Layout(width=window_width, margin='0px', padding='0px', overflow='visible'))) 
            elif i == len(windows) - 1:
                windows_with_page_buttons.append( widgets.VBox([refresh, window, submit, prev_pg_formatted], layout=widgets.Layout(width=window_width, margin='0px', padding='0px', overflow='visible'))) 
            else:
                windows_with_page_buttons.append( widgets.VBox([refresh, window, pg_group], layout=widgets.Layout(width=window_width, margin='0px', padding='0px', overflow='visible')) )
                
        windows = windows_with_page_buttons
        CURRENT_WINDOW = widgets.VBox([windows[0]])

        def refresh_window(_):
            BinaryDisplay._main_output.clear_output()
            BinaryDisplay.display()

        def get_pg(button):
            nonlocal CURRENT_WINDOW_INDEX
            nonlocal CURRENT_WINDOW
            nonlocal windows
            
            if button == next_pg:
                CURRENT_WINDOW_INDEX += 1
            else:
                CURRENT_WINDOW_INDEX -= 1
                
            CURRENT_WINDOW.children = [windows[CURRENT_WINDOW_INDEX]] 

        def start_proccessing(_):
            if len(BinaryDisplay.entries) < 1 and BinaryDisplay.file is None:
                print('No Targets Selected')
                return 

            if BinaryDisplay.author == None:
                print('No Author Selected')
                return

            if True not in [val for val in BinaryDisplay.processes.values()] + [val for val in BinaryDisplay.processes['Animate'].values()]:
                print('No Processes Selected')
                return 
                
            BinaryDisplay._main_output.clear_output()
            BinaryDisplay._proccess()

        refresh.on_click(refresh_window)
        next_pg.on_click(get_pg)
        prev_pg.on_click(get_pg)
        submit.on_click(start_proccessing)

        with BinaryDisplay._main_output:
            display(CURRENT_WINDOW)
        
        display(BinaryDisplay._main_output)
        return 







