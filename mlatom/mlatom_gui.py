#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! mlatom_gui: GUI for MLatom                                                ! 
  ! Implementations by: Bao-Xin Xue                                           ! 
  !---------------------------------------------------------------------------! 
'''

from subprocess import PIPE, Popen
from sys import stdout
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import tkinter.messagebox as messagebox
from typing import Any, Callable, List, Dict, Tuple, Union
from dataclasses import dataclass
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from io import TextIOWrapper
import threading

container = Union[Frame, Tk, LabelFrame, Toplevel]

def judge_value_type(value: str) -> type:
    pass

class GuiValueException(BaseException): pass


class TkItem():
    label = 'label'.lower()
    entry = 'entry'.lower()
    radioButton = 'radioButton'.lower()
    checkButton = 'checkButton'.lower()
    labelFrame = 'LabelFrame'.lower()
    noteBook = 'noteBook'.lower()
    combobox = 'combobox'.lower()
    fileSelector = 'fileSelector'.lower()


class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info', additional_x=0):
        self.waittime = 500     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None
        self.dx = additional_x

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x + self.dx, y))   # todo: modified width of ToolTip
        label = Label(self.tw, text=self.text, justify='left',
                        background="#ffffff", relief='solid', borderwidth=1,
                        wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()


class Widget(ABC):
    type: List[type]
    created: bool = False
    @property
    @abstractmethod
    def out_string(self) -> str: pass

    @abstractmethod
    def create_widget(self, root: container) -> None: pass

@dataclass
class HorizontalFrame(Widget):
    widgets: List[Widget]
    description: str = ''
    tool_tip_dx: int = 0
    
    def create_widget(self, root: container) -> None:
        self.created = True
        grid_frame = Frame(root)
        for idx, w in enumerate(self.widgets):
            f = Frame(grid_frame)
            w.create_widget(f)
            f.grid(row=0, column=idx)
        if self.description:
            CreateToolTip(grid_frame, self.description, self.tool_tip_dx)
        grid_frame.pack()
    
    @property
    def out_string(self) -> str:
        tmp = ''
        for w in self.widgets:
            w_str = w.out_string.strip()
            if w_str:
                tmp += w_str + '\n'
        tmp = tmp.strip()
        return tmp


@dataclass
class VerticalFrame(Widget):
    widgets: List[Widget]
    description: str = ''
    tool_tip_dx: int = 0
    
    def create_widget(self, root: container) -> None:
        self.created = True
        grid_frame = Frame(root)
        for idx, w in enumerate(self.widgets):
            f = Frame(grid_frame)
            w.create_widget(f)
            f.grid(row=idx, column=0)
        if self.description:
            CreateToolTip(grid_frame, self.description, self.tool_tip_dx)
        grid_frame.pack()
    
    @property
    def out_string(self) -> str:
        tmp = ''
        for w in self.widgets:
            w_str = w.out_string.strip()
            if w_str:
                tmp += w_str + '\n'
        tmp = tmp.strip()
        return tmp


@dataclass
class TopLevelWindowBtn(Widget, Toplevel):
    btn_text: str
    win_title: str = ''
    widgets: List[Widget] = None
    # def __init__(self, btn_text: str, win_title: str='', widgets: List[Widget]=[]) -> None:
    #     self.btn_text = btn_text
    #     self.win_title = win_title
    #     self.widgets = widgets
    
    def create_widget(self, root: container) -> None:
        self.created = True
        self.create_window()
        btn = Button(root, text=self.btn_text, command=self.show)
        btn.pack()
        
    def create_window(self):
        super().__init__()
        if self.win_title:
            self.title(self.win_title)
        self.protocol('WM_DELETE_WINDOW', self.hide)
        f = Frame(self)
        for w in self.widgets:
            w.create_widget(f)
        f.pack()
        self.hide()

    def show(self) -> None:
        self.update()
        self.deiconify()

    @property
    def out_string(self) -> str:
        tmp = ''
        for w in self.widgets:
            out = w.out_string.strip()
            if out:
                tmp += out + '\n'
        tmp = tmp.strip()
        return tmp

    def hide(self) -> None:
        self.withdraw()
            

class MyLabel(Widget):
    def __init__(self, text: str, description: str='') -> None:
        self.text = text
        self.description = description
    
    def create_widget(self, root: container) -> None:
        self.created = True
        label = Label(root, text=self.text)
        label.pack(fill=BOTH)
        if self.description:
            CreateToolTip(label, self.description)
    
    @property
    def out_string(self) -> str: return ''


@dataclass
class MultiEntry(Widget):
    fmt_str: str
    entry_num: int
    description: str
    present_str: str = ''

    def __post_init__(self):
        self.entry_list: List[Entry] = []

    def create_widget(self, root: container) -> None:
        self.created = True
        frame = Frame(root)
        kw_label = Label(frame, text=self.present_str)
        kw_label.grid(row=0, column=0)
        CreateToolTip(kw_label, self.description)
        for i in range(1, self.entry_num + 1):
            entry = Entry(frame)
            entry.grid(row=0, column=i)
            self.entry_list.append(entry)
            CreateToolTip(entry, self.description)
        frame.pack()
    
    @property
    def out_string(self) -> str:
        value_list: List[str] = []
        for entry in self.entry_list:
            value_list.append(entry.get().strip())
        has_content = False
        for value in value_list:
            if value:
                has_content = True
                break
        if has_content:
            return self.fmt_str % tuple(value_list)
        return ''


class KW_Entry(Widget):
    placeholder_style = "Placeholder.TEntry"
    input_style = "TEntry"
    def __init__(self, kw_name: str, description: str, present_str: str='') -> None:
        self.kw_name = kw_name
        self.description = description
        self.present_str = present_str
        self.entry: Entry
        self.added_kw = False
        # self.create_widget()
    
    def create_widget(self, root: container):
        self.created = True
        frame = Frame(root)
        if self.present_str:
            kw_label = Label(frame, text=self.present_str)
        else:
            kw_label = Label(frame, text=self.kw_name)
        if not self.added_kw:
            self.description = self.description + f'\nkeyword: {self.kw_name}'
            self.added_kw = True
        kw_label.pack(fill=Y, side='left')
        # super().__init__(frame)
        self.entry = Entry(frame)
        CreateToolTip(kw_label, self.description)
        CreateToolTip(self.entry, self.description)
        self.entry.pack(fill=Y)
        frame.pack()
        # self.insert('0', self.description)
        # self['foreground'] = '#d5d5d5'
        # self.bind('<FocusIn>', self._clear_placeholder)
        # self.bind('<FocusOut>', self._add_placeholder)

    @property
    def out_string(self) -> str:
        content: str = self.entry.get().strip()
        if content:
            return f'{self.kw_name}={content.strip()}'
        else:
            return ''
    
    def _clear_placeholder(self, e):
        # if self['style'] == self.placeholder_style:
        if self.entry['foreground'] == '#d5d5d5':
            self.entry.delete('0', 'end')
            # self['style'] = self.input_style
            self.entry['foreground'] = '#ffffff'
    
    def _add_placeholder(self, e):
        if not self.entry.get():
            self.entry.insert('0', self.description)
            # self['style'] = self.placeholder_style
            self.entry['foreground'] = '#d5d5d5'


@dataclass
class MyMultiLineText(Widget):
    description: str

    def __post_init__(self):
        self.text: Text
    
    def create_widget(self, root: container) -> None:
        self.created = True
        # output window
        f = Frame(root, height=15)
        self.text = Text(f, width=60, height=15)   # text = Text(f, width=80, height=60)
        out_scroll = Scrollbar(f)
        self.text.pack(side=LEFT, fill=Y)
        out_scroll.pack(side=RIGHT, fill=Y)
        out_scroll.config(command=self.text.yview)
        self.text.config(yscrollcommand=out_scroll.set)
        f.pack()
        CreateToolTip(f, self.description)
    
    @property
    def out_string(self) -> str:
        content: str = self.text.get('1.0', END).strip()
        return content if content else ''


class EntryOfFileBtn(Widget):
    def __init__(self, kw_name:str, description: str, present_str: str='') -> None:
        self.kw_name = kw_name
        self.description = description
        self.entry_text: str = ''
        self.present_str = present_str
        self.entry = None
        # self.create_widget()
        
    def create_widget(self, root: container):
        self.created = True
        frame = Frame(root)
        if self.present_str:
            kw_label = Label(frame, text=self.present_str)
        else:
            kw_label = Label(frame, text=self.kw_name)
        self.description = self.description + f'\nkeyword: {self.kw_name}'
        kw_label.pack(fill=Y, side='left')
        self.entry = Entry(frame)
        btn = Button(frame, text='choose file', command=self.select_file)
        btn.pack(fill=Y, side='right')
        self.entry.pack(fill=Y)
        CreateToolTip(kw_label, self.description)
        CreateToolTip(self.entry, self.description)
        CreateToolTip(btn, self.description)
        frame.pack()
    
    def select_file(self):
        path: str = askopenfilename()
        self.entry_text = os.path.relpath(path)
        self.entry.delete('0', 'end')
        self.entry.insert('0', self.entry_text)
    
    @property
    def out_string(self) -> str:
        content: str = self.entry.get().strip()
        if content:
            return f'{self.kw_name}={content}'
        else:
            return ''


class Check_Button(Widget):
    # variable
    def __init__(self, title: str, items: List[str], description: List[str]=[], present_str: List[str]=[]) -> None:
        self.title = title
        self.items = items
        self.description = description
        self.present_str = present_str
        self._value_list: List[BooleanVar] = []
    
    def create_widget(self, root: container):
        self.created = True
        if self.title:
            frame = LabelFrame(root, text=self.title)
        else:
            frame = root
        for idx, kw_name in enumerate(self.items):
            v = BooleanVar(frame)
            if self.present_str:
                show_str = self.present_str[idx]
            else:
                show_str = kw_name
            cb = Checkbutton(frame, text=show_str, variable=v,
                             onvalue=True, offvalue=False)
            try:
                CreateToolTip(cb, self.description[idx] + f'\nkeyword: {kw_name}')
            except:
                messagebox.showerror(f'description on check_button {self.title} is not match with the number of item')
            cb.pack()
            self._value_list.append(v)
        if self.title:
            frame.pack()

    @property
    def out_string(self) -> str:
        true_list: List[str] = []
        for item, bool_var in zip(self.items, self._value_list):
            if bool_var.get():
                true_list.append(item)
        return '\n'.join(true_list).strip()


class MyNotebook(Widget):
    def __init__(self, nb_items: Dict[str, List[Widget]]) -> None:
        self.nb_items = nb_items
    
    def create_widget(self, root: container) -> None:
        self.created = True
        nb = Notebook(root)
        for nb_name, widgets in self.nb_items.items():
            frame = Frame(nb)
            for sub_widget in widgets:
                sub_widget.create_widget(frame)
            nb.add(frame, text=nb_name)
        # nb.bind("<<NotebookTabChanged>>", self.tab_resize)
        nb.pack()
    
    # def tab_resize(self, event):
    #     event.widget.update_idletasks()
    #     tab = event.widget.nametowidget(event.widget.select())
    #     event.widget.configure(height=tab.winfo_reqheight())
    
    @property
    def out_string(self) -> str:
        tmp = ''
        for _, widgets in self.nb_items.items():
            for w in widgets:
                out = w.out_string.strip()
                if out:
                    tmp += out + '\n'
        return tmp.strip()


class MyLabelFrame(Widget):
    def __init__(self, name: str, widgets: List[Widget], default_kw: str='', description: str='') -> None:
        self.name = name
        self.widgets = widgets
        self.default_kw = default_kw
        self.description = description
    
    def create_widget(self, root: container) -> None:
        self.created = True
        lf = LabelFrame(root, text=self.name)
        for w in self.widgets:
            w.create_widget(lf)
        lf.pack()
        if self.description:
            CreateToolTip(lf, text=self.description)
    
    @property
    def out_string(self) -> str:
        tmp = ''
        for w in self.widgets:
            out = w.out_string.strip()
            if out:
                tmp += out + '\n'
        tmp = tmp.strip()
        if tmp and self.default_kw:
            return self.default_kw + '\n' + tmp
        return tmp


class MyComboBox(Widget):
    def __init__(self, kw_name: str, candidates: List[str], description: str, compulsory: bool=False, present_str: str='') -> None:
        self.kw_name = kw_name
        self.candidates = candidates
        self.description = description
        self.compulsory = compulsory
        self.present_str = present_str
    
    def create_widget(self, root: container) -> None:
        self.created = True
        f = Frame(root)
        if self.present_str:
            label = Label(f, text=self.present_str)
        else:
            label = Label(f, text=self.kw_name)
        label.pack(fill=Y, side='left')
        self.value = StringVar(f)
        cbx = Combobox(f, textvariable=self.value)
        cbx["values"] = self.candidates
        cbx.pack()
        f.pack()
        if self.kw_name:
            CreateToolTip(cbx, self.description + f'\nkeyword: {self.kw_name}')
        else:
            CreateToolTip(cbx, self.description)
    
    @property
    def out_string(self):
        selected = self.value.get().strip()
        if self.compulsory and selected == '':
            messagebox.showerror('error', f'You Must Choose a Value ({", ".join(self.candidates)}) in Combobox Widget "{self.kw_name}".')
            raise GuiValueException
        if selected:
            if self.kw_name:
                return f'{self.kw_name}={selected}'
            else:
                return selected
        else:
            return ''


class OutputBtn(Widget):
    output_window: Toplevel = None
    err_window: Toplevel = None
    output_text_widget: Text = None
    err_text_widget: Text = None
    def __init__(self, widgets: Union[List[Widget], List[List[Widget]]], default_keyword: str='') -> None:
        self.widgets = widgets
        self.default_kw = default_keyword.strip()
        self.file_name_entry: Entry
        self.run_lock = False
    
    def create_widget(self, root: container) -> None:
        self.created = True
        # write input file
        f = Frame(root)
        label = Label(f, text='input file name: ')
        label.grid(row=0, column=0, sticky=E)
        self.file_name_entry = Entry(f)
        self.file_name_entry.grid(row=0, column=1, sticky=W)
        self.file_name_entry.insert('0', 'mlatom.inp')
        out_btn = Button(f, text='generate input file', command=self.output_btn_command)
        out_btn.grid(row=1, column=0, sticky=W+E)
        calc_btn = Button(f, text='calculate', command=lambda: self.thread_it(self.calc_command, args={}))
        # calc_btn = Button(f, text='calculate', command=self.calc_command)
        calc_btn.grid(row=1, column=1, sticky=W+E)
        f.pack()
        # output window
        if not self.__class__.output_window:
            self.create_window()
        
    def create_window(self):
        # output window
        output_window = Toplevel()
        self.__class__.output_window = output_window
        output_window.title('MLatom Output')
        output_window.protocol('WM_DELETE_WINDOW', output_window.withdraw)
        f_out = Frame(output_window)
        out_text = Text(f_out)   # text = Text(f, width=80, height=60)
        self.__class__.output_text_widget = out_text
        out_scroll = Scrollbar(f_out)
        out_text.pack(side=LEFT, fill=Y)
        out_scroll.pack(side=RIGHT, fill=Y)
        out_scroll.config(command=out_text.yview)
        out_text.config(yscrollcommand=out_scroll.set)
        f_out.pack()
        output_window.withdraw()
        

        # error window
        err_window = Toplevel()
        self.__class__.err_window = err_window
        err_window.title('MLatom Error')
        err_window.protocol('WM_DELETE_WINDOW', err_window.withdraw)
        f_err = Frame(err_window)
        err_text = Text(f_err)
        self.__class__.err_text_widget = err_text
        err_scroll = Scrollbar(f_err)
        err_text.pack(side=LEFT, fill=Y)
        err_scroll.pack(side=RIGHT, fill=Y)
        out_scroll.config(command=err_text.yview)
        err_text.config(yscrollcommand=err_scroll.set)
        f_err.pack()
        err_window.withdraw()
    
    @staticmethod
    def get_out_string(w: Widget) -> str:
        out_string = w.out_string.strip()
        return out_string + '\n' if out_string else ''
    
    @staticmethod
    def calc_thread_task(text_widget: Text, output_window: Toplevel, file_name: str) -> Tuple[str, str]:
        try:
            from MLatom.shell_cmd import mlatom
            run = Popen(f'mlatom {file_name}', stderr=PIPE, stdout=PIPE, shell=True)
        except:
            run = Popen(f'MLatom.py {file_name}', stderr=PIPE, stdout=PIPE, shell=True)
        whole_out = ''
        whole_err = ''

        for line in TextIOWrapper(run.stdout, encoding='utf-8'):
            whole_out += line
            text_widget.insert(END, line)
            output_window.update()
        for line in TextIOWrapper(run.stderr, encoding='utf-8'):
            whole_err += line
        
        out_file = file_name + '.out'
        with open(out_file, 'w') as f:
            f.write(whole_out)
        if whole_err.strip():
            err_file = file_name + '.err'
            with open(err_file, 'w') as f:
                f.write(whole_err)
            messagebox.showerror('fail', f'calculation failed, see {out_file} and {err_file} for more detail.')
            output_window.title(f'MLatom {file_name} calculation fail.')
        else:
            output_window.title(f'MLatom {file_name} calculation ended.')
            messagebox.showinfo('success', f'calculation end successfully, see {out_file} for more detail.')
        output_window.update()
        output_window.deiconify()
        # return whole_out, whole_err
    
    @staticmethod
    def add_to_text_widget(text_widget: Text, string: str):
        text_widget.insert(END, string)
    
    def thread_it(self, func: Callable[..., Any], args: Any):
        if self.run_lock:
            messagebox.showinfo('warnning', f'MLatom is running, please wait current task finished to start a new task.')
        else:
            self.run_lock = True
            t = threading.Thread(target=func, args=args)
            t.start()
            # t.join()

    def calc_command(self):
        # raise NotImplementedError
        # messagebox.showerror('error', 'This function still not implemented!')
        output_text_widget = self.__class__.output_text_widget
        output_text_widget.delete('1.0', 'end')
        self.output_btn_command()
        messagebox.showinfo('info', 'MLatom is running.')
        file_name: str = self.file_name_entry.get().strip()
        output_window = self.__class__.output_window
        output_window.title(f'MLatom {file_name} is running, please wait.')
        output_window.update()
        output_window.deiconify()
        # calc_thread = threading.Thread(target=self.calc_thread_task, args=(output_text_widget, output_window, file_name))
        # calc_thread.start()
        # calc_thread.join()
        try:
            from MLatom.shell_cmd import mlatom
            run = Popen(f'mlatom {file_name}', stderr=PIPE, stdout=PIPE, shell=True)
        except:
            run = Popen(f'MLatom.py {file_name}', stderr=PIPE, stdout=PIPE, shell=True)
        whole_out = ''
        whole_err = ''

        for line in TextIOWrapper(run.stdout, encoding='utf-8'):
            whole_out += line
            output_text_widget.insert(END, line)
            # self.add_to_text_widget(output_text_widget, line)
            output_window.update()
        for line in TextIOWrapper(run.stderr, encoding='utf-8'):
            whole_err += line

        out_file = file_name + '.out'
        with open(out_file, 'w') as f:
            f.write(whole_out)
        if whole_err.strip():
            err_file = file_name + '.err'
            with open(err_file, 'w') as f:
                f.write(whole_err)
            messagebox.showerror('fail', f'calculation failed, see {out_file} and {err_file} for more detail.')
            output_window.title(f'MLatom {file_name} calculation fail.')
            self.__class__.err_text_widget.delete('1.0','end')
            self.__class__.err_text_widget.insert(END, whole_err)
            self.__class__.err_window.update()
            self.__class__.err_window.deiconify()
        else:
            output_window.title(f'MLatom {file_name} calculation ended.')
            messagebox.showinfo('success', f'calculation end successfully, see {out_file} for more detail.')
        output_window.update()
        output_window.deiconify()
        self.run_lock = False
        
    def out_to_inp_file(self, input_file: str='mlatom.inp'):
        tmp: str = ''
        try:
            if self.default_kw:
                tmp = self.default_kw + '\n'
            for w in self.widgets:
                if isinstance(w, Widget):
                # if type(w) is Widget:
                    tmp += self.get_out_string(w)
                elif type(w) is list:
                    for sub_w in w:
                        tmp += self.get_out_string(sub_w)
        except GuiValueException:
            messagebox.showerror('fail', 'Value inputed error, input file not generated.')
        try:
            with open(input_file, 'w') as f:
                f.write(tmp)
            messagebox.showinfo(f'success', f'MLatom input file has been writen into "{input_file}".')
        except:
            messagebox.showerror(f'fail', f'writing "{input_file}" failed.')
    
    def output_btn_command(self):
        file_name: str = self.file_name_entry.get().strip()
        self.out_to_inp_file(file_name)
    
    @property
    def out_string(self) -> str:
        return ''


class GUI(Tk):
    def add_item(self, root: container, item: Union[Widget, List[Widget]]):
        if isinstance(item, Widget):
            item.create_widget(root)
        elif type(item) is list:
            for each_item in item:
                each_item.create_widget(root)


def main():
    ##### ML interface and detail configuration #####
    KRR_param: List[Widget] = [
        HorizontalFrame([

        MyLabelFrame('kernel & hyperparameter', [
            MyLabelFrame('kernel', [
                MyComboBox('kernel', ['Gaussian', 'Laplacian', 'Linear', 'Polynomial', 'exponential', 'Matern'], 
                    'default kernel parameter table\n' \
                    'kernel  sigma  NlgSigma  lgSigmaL  lgSigmaH\n' \
                    'Gaussian  100.0      11       2.0       9.0\n' \
                    'Laplacian 800.0      11       5.0      12.0\n' \
                    'exponential 800.0    11       5.0      12.0\n' \
                    'Matern    100.0      11       2.0       9.0\n'
                ),
                MyLabelFrame('Matern kernel option', [
                    KW_Entry('nn', 'n in the Matern kernel (nu = n + 1/2)\nin default nn=2')
                ]),
            ]),
            MyLabelFrame('lambda', [
                KW_Entry('lambda', 'regularization hyperparameter R [0 by default] \n can set it as opt'),
                Check_Button('', ['lambda=opt'], ['requests optimization of lambda on a log grid'], ['optimize lambda value']),
                KW_Entry('NlgLambda', 'N points on a logarithmic grid [6 by default]'),
                KW_Entry('lgLambdaL', 'lowest  value of log2(lambda) [-16.0 by default]'),
                KW_Entry('lgLambdaH', 'highest value of log2(lambda) [ -6.0 by default]'),
            ]),
            MyLabelFrame('sigma', [
                KW_Entry('sigma', 'length scale [default values: 100 (Gaussian  & Matern) 800 (Laplacian & exponential)]'),
                Check_Button('', ['sigma=opt'], ['requests optimization of sigma on a log grid'], ['optimize sigma value']),
                KW_Entry('NlgSigma', 'N points on a logarithmic grid [11 by default]'),
                KW_Entry('lgSigmaL', 'lowest  value of log2(lambda) [default values:  2.0 (Gaussian  & Matern) 5.0 (Laplacian & exponential)]'),
                KW_Entry('lgSigmaH', 'highest value of log2(lambda) [default values:  9.0 (Gaussian  & Matern) 12.0 (Laplacian & exponential)]'),
            ]),
        ]),
        MyLabelFrame('calculation option', [
            MyComboBox('matDecomp', ['Cholesky', 'LU', 'Bunch-Kaufman'], 'type of matrix decomposition'),
            MyLabelFrame('permInvKernel', [
                KW_Entry('Nperm', 'number of permutations N (with XfileIn argument)'),
                Check_Button('', ['molDescrType=permuted'], ['with XYZfile argument']),
            ], default_kw='permInvKernel', description='permutationally invariant kernel'),
            Check_Button('other option', ['invMatrix', 'refine', 'on-the-fly', 'benchmark', 'debug'], 
                        ['invert matrix', 'refine solution matrix', 'on-the-fly calculation of kernel matrix elements for validation, by default it is false and those elements are stored',
                        'additional output for benchmarking', 'additional output for debugging']),
            MyLabelFrame('hyperparameter optimization', [
                MyComboBox('minimizeError', ['RMSE', 'MAE'], 'type S of error to minimize'),
                KW_Entry('lgOptDepth', 'depth of log grid optimization N [3 by default]'),
            ]),
        ]),

        ]),
        MyLabelFrame('molecular descriptor option', [
            HorizontalFrame([
                MyComboBox('molDescriptor', ['RE', 'CM'],
                            '      RE [default]             vector {Req/R}, where R is internuclear distance\n'
                            '      CM                       Coulomb matrix'
                ),
                MyComboBox('molDescrType', ['sorted', 'unsorted', 'permuted'],
                    '      sorted                   default for molDescrType=CM\n'
                    '                               sort by:\n'
                    '                                 norms of CM matrix (for molDescrType=CM)\n'
                    '                                 nuclear repulsions (for molDescrType=RE)\n'
                    '      unsorted                 default for molDescrType=RE\n'
                    '      permuted                 molecular descriptor with all atom permutations'
                ),
            ]),
            MyLabelFrame('additional option for molDescrType=sorted and molDescriptor=RE', [
                KW_Entry('XYZsortedFileOut', 'file with sorted XYZ coordinates')
            ]),
            MyLabelFrame('additional option for molDescrType=permuted', [
                KW_Entry('permInvGroups', 
                        'permutationally invariant groups S\n'
                        'e.g. for water dimer permInvGroups=1,2,3-4,5,6\n'
                        'permute water molecules (atoms 1,2,3 and 5,6,7)'
                ),
                KW_Entry('permInvNuclei', 
                        'permutationally invariant nuclei S\n'
                        'e.g.permInvNuclei=2-3.5-6\n'
                        'will permute atoms 2,3 and 6,7'
                )
            ])
        ]),
    ]
    KRR_btn = TopLevelWindowBtn('KRR', 'KRR task option', KRR_param)
    # todo: check whether MLprog is duplicated   
    sgdml_param: List[Widget] = [ 
        MyLabelFrame('sGDML option', [ 
            MyComboBox('MLmodelType', candidates=['sGDML', 'GDML'], description='requests model, default: sGDML'),
            MyComboBox('sgdml.gdml', ['True', 'False'], 'use GDML instead of sGDML, default=False'),
            MyComboBox('sgdml.cprsn', ['True', 'False'], 'compress kernel matrix along symmetric degrees of freedom, default=False'),
            MyComboBox('sgdml.no_E', ['True', 'False'], 'do not predict energies, default=False'),
            MyComboBox('sgdml.E_cstr', ['True', 'False'], 'include the energy constraints in the kernel, default=False'),
            KW_Entry('sgdml.s', 'sgdml.s=<s1>[,<s2>[,...]]  set hyperparameter sigma=<start>:[<step>:]<stop>'),
        ], default_kw='MLprog=sGDML'),
    ]
    sgdml_btn = TopLevelWindowBtn('sGDML', 'sGDML option', sgdml_param)
    gap_param: List[Widget] = [ 
        MyLabelFrame('GAP option', [ 
            MyMultiLineText('gapfit.xxx=x   \nxxx could be any option for gap_fit Note that at_file and gp_file are not required'),
            MyMultiLineText('gapfit.gap.xxx=x  \nxxx could be any option for gap'),
            MultiEntry('gapfit.default_sigma={%s,%s,%s,%s}', 4, 'sigmas for energies, forces, virals, Hessians, default is {0.0005,0.001,0,0}', 'gapfit.default_sigma'),
            KW_Entry('gapfit.e0_method', 'method for determining e0, default is average'),
            KW_Entry('gapfit.gap.type', 'descriptor type, default is soap'),
            KW_Entry('gapfit.gap.l_max', 'max number of angular basis functions, default is 6'),
            KW_Entry('gapfit.gap.n_max', 'max number of radial  basis functions, default is 6'),
            KW_Entry('gapfit.gap.atom_sigma', 'Gaussian smearing of atom density hyperparameter, default is 0.5'),
            KW_Entry('gapfit.gap.zeta', 'hyperparameter for kernel sensitivity, default is 4'),
            KW_Entry('gapfit.gap.cutoff', 'cutoff radius of local environment, default is 6.0'),
            KW_Entry('gapfit.gap.cutoff_transition_width', 'cutoff transition width, default is 0.5'),
            KW_Entry('gapfit.gap.delta', 'hyperparameter delta for kernel scaling, default is 1'),
        ], default_kw='MLprog=GAP\nMLmodelType=GAP-SOAP'),
    ]
    gap_btn = TopLevelWindowBtn('GAP', 'GAP option', gap_param)
    PhysNet_param: List[Widget] = [ 
        MyLabelFrame('PhysNet option', [
            KW_Entry('physnet.num_features', 'number of input features, default is 128'),
            KW_Entry('physnet.num_basis', 'number of radial basis functions, default is 64'),
            KW_Entry('physnet.num_blocks', 'number of stacked modular building blocks, default is 5'),
            KW_Entry('physnet.num_residual_atomic', 'number of residual blocks for atom-wise refinements, default is 2'),
            KW_Entry('physnet.num_residual_interaction', 'number of residual blocks for refinements of proto-message, default is 3'),
            KW_Entry('physnet.num_residual_output', 'number of residual blocks in output blocks, default is 1'),
            KW_Entry('physnet.cutoff', 'cutoff radius for interactions in the neural network, default is 10.0'),
            KW_Entry('physnet.seed', 'random seed, default is 42'),
            KW_Entry('physnet.max_steps', 'max steps to perform in training, default is 10000000'),
            KW_Entry('physnet.learning_rate', 'starting learning rate, default is 0.0008'),
            KW_Entry('physnet.decay_steps', 'decay steps, default is 10000000'),
            KW_Entry('physnet.decay_rate', 'decay rate for learning rate, default is 0.1'),
            KW_Entry('physnet.batch_size', 'training batch size, default is 12'),
            KW_Entry('physnet.valid_batch_size', 'validation batch size, default is 2'),
            KW_Entry('physnet.force_weight', 'weight for force, default is 52.91772105638412'),
            KW_Entry('physnet.charge_weight', 'weight for charge, default is 0'),
            KW_Entry('physnet.dipole_weight', 'weight for dipole, default is 0'),
            KW_Entry('physnet.summary_interval', 'interval for summary, default is 5'),
            KW_Entry('physnet.validation_interval', 'interval for validation, default is 5'),
            KW_Entry('physnet.save_interval', 'interval for model saving, default is 10'),
        ], default_kw='MLprog=PhysNet\nMLmodelType=PhysNet'),
    ]
    PhysNet_btn = TopLevelWindowBtn('PhysNet', 'PhysNet option', PhysNet_param)
    deepmd_param: List[Widget] = [ 
        MyLabelFrame('DeepMD-kit option', [ 
            MyComboBox('MLmodelType', ['DeepPot-SE', 'DPMD'], 'requests model, default is DeepPot-SE'),
            MyMultiLineText('deepmd.xxx.xxx=X   \nspecify arguments for DeePMD, follows DeePMD-kit\'s json input file structure'),
            KW_Entry('deepmd.training.stop_batch', 'number of batches to be trained before stopping, default is 4000000'),
            KW_Entry('deepmd.training.batch_size', 'size of each batch, default is 32'),
            KW_Entry('deepmd.learning_rate.start_lr', 'initial learning rate, default is 0.001'),
            KW_Entry('deepmd.learning_rate.decay_steps', 'number of batches for one decay, default is 4000'),
            KW_Entry('deepmd.learning_rate.decay_rate', 'decay rate of each decay, default is 0.95'),
            KW_Entry('deepmd.model.descriptor.rcut', 'cutoff radius for local environment, default is 6.0'),
            KW_Entry('deepmd.model.fitting_net.neuron', 'NN structure of fitting network, default is 80,80,80'),
            EntryOfFileBtn('deepmd.input', 'file with DeePMD input parameters in json format (as a template)'),
        ], default_kw='MLprog=DeePMD-kit')
    ]
    deepmd_btn = TopLevelWindowBtn('DeepMD', 'DeepMD option', deepmd_param)
    torch_ani_param: List[Widget] = [ 
        MyLabelFrame('TorchANI option', [
            KW_Entry('ani.batch_size', 'batch size, default is 8'),
            KW_Entry('ani.max_epochs', 'max epochs, default is 10000000'),
            KW_Entry('ani.early_stopping_learning_rate', 'learning rate that triggers early-stopping, default is 0.00001'),
            KW_Entry('ani.force_coefficient', 'weight for force, default is 0.1'),
            KW_Entry('ani.Rcr', 'radial cutoff radius, default is 5.2'),
            KW_Entry('ani.Rca', 'angular cutoff radius, default is 3.5'),
            KW_Entry('ani.EtaR', 'radial smoothness in radial part, default is 1.6'),
            KW_Entry('ani.ShfR', 'radial shifts in radial part, default is 0.9,1.16875,1.4375,1.70625,1.975,2.24375,2.5125,2.78125,3.05,3.31875,3.5875,3.85625,4.125,4.9375,4.6625,4.93125'),
            KW_Entry('ani.Zeta', 'angular smoothness, default is 32'),
            KW_Entry('ani.ShfZ', 'angular shifts, default is 0.19634954,0.58904862,0.9817477,1.3744468,1.7671459,2.1598449,2.552544,2.9452431'),
            KW_Entry('ani.EtaA', 'radial smoothness in angular part, default is 8'),
            KW_Entry('ani.ShfA', 'radial shifts in angular part, default is 0.9,1.55,2.2,2.85'),
            KW_Entry('ani.Neuron_l1', 'number of neurons in layer 1, default is 160'),
            KW_Entry('ani.Neuron_l2', 'number of neurons in layer 2, default is 128'),
            KW_Entry('ani.Neuron_l3', 'number of neurons in layer 3, default is 96'),
            KW_Entry('ani.AF1', 'acitivation function for layer 1, default is \'CELU\''),
            KW_Entry('ani.AF2', 'acitivation function for layer 2, default is \'CELU\''),
            KW_Entry('ani.AF3', 'acitivation function for layer 3, default is \'CELU\''),
        ], default_kw='MLprog=TorchANI\nMLmodelType=ANI'),
    ]
    torch_ani_btn = TopLevelWindowBtn('TorchANI', 'TorchANI option', torch_ani_param)
    hyperopt_param: List[Widget] = [
        MyLabelFrame('Arguments for hyperopt.xx():', [
            MultiEntry('hyperopt.uniform(%s,%s)', 2, 'hyperopt.uniform(lb,ub)    linear search space from lb to ub', 'hyperopt.uniform'),
            MultiEntry('hyperopt.loguniform(%s,%s)', 2, 'hyperopt.loguniform(lb,ub) logarithmic search space, base 2', 'hyperopt.loguniform'),
            MultiEntry('hyperopt.quniform(%s, %s, %s)', 3, 'hyperopt.quniform(lb,ub,q) discrete linear space, rounded by q lb is lower bound, ub is upper bound', 
                        'hyperopt.quniform')
        ]),
        MyLabelFrame('Other Arguments', [
            KW_Entry('hyperopt.max_evals', 'max number of search attempts [8 by default]'),
            MyComboBox('hyperopt.losstype', ['geomean', 'weighted'], 
                        description='hyperopt.losstype=S      type of loss used in optimization\n'
                                    '  geomean [default]      geometric mean\n'
                                    '  weighted               weight for gradients, defined by'),
            KW_Entry('hyperopt.w_grad', '[0.1 by default]'),
        ]),
    ]
    hyperopt_btn = TopLevelWindowBtn('hyperopt', 'hyperopt option', hyperopt_param)
    ML_algorithm = HorizontalFrame([KRR_btn, sgdml_btn, gap_btn, PhysNet_btn, deepmd_btn, torch_ani_btn, hyperopt_btn],
        '  Supported ML model types and default programs:\n' \
        '        \n' \
        '  +-------------+----------------+\n' \
        '  | MLmodelType | default MLprog |\n' \
        '  +-------------+----------------+\n' \
        '  | KREG        | MLatomF        |\n' \
        '  +-------------+----------------+\n' \
        '  | sGDML       | sGDML          |\n' \
        '  +-------------+----------- ----+\n' \
        '  | GAP-SOAP    | GAP            |\n' \
        '  +-------------+----------------+\n' \
        '  | PhysNet     | PhysNet        |\n' \
        '  +-------------+----------------+\n' \
        '  | DeepPot-SE  | DeePMD-kit     |\n' \
        '  +-------------+----------------+\n' \
        '  | ANI         | TorchANI       |\n' \
        '  +-------------+----------------+\n' \
        '     \n' \
        '  Supported interfaces with default and tested ML model types:\n' \
        '    \n' \
        '  +------------+----------------------+\n' \
        '  | MLprog     | MLmodelType          |\n' \
        '  +------------+----------------------+\n' \
        '  | MLatomF    | KREG [default]       |\n' \
        '  |            | see                  |\n' \
        '  |            | MLatom.py KRR help   |\n' \
        '  +------------+----------------------+\n' \
        '  | sGDML      | sGDML [default]      |\n' \
        '  |            | GDML                 |\n' \
        '  +------------+----------------------+\n' \
        '  | GAP        | GAP-SOAP             |\n' \
        '  +------------+----------------------+\n' \
        '  | PhysNet    | PhysNet              |\n' \
        '  +------------+----------------------+\n' \
        '  | DeePMD-kit | DeepPot-SE [default] |\n' \
        '  |            | DPMD                 |\n' \
        '  +------------+----------------------+\n' \
        '  | TorchANI   | ANI [default]        |\n' \
        '  +------------+----------------------+\n',
        tool_tip_dx=200
    )

    ##### machine learning task #####
    delta_learn_widget: Widget = MyLabelFrame('delta learning', [
                KW_Entry('Yb', 'file with data obtained with baseline method'),
                KW_Entry('Yt', 'file with data obtained with target method'),
                KW_Entry('YestT', 'file with ML estimations of target method'),
                KW_Entry('YestFile', 'file with ML corrections to  baseline method')
            ], default_kw='deltaLearn', description='when this frame is filled,\n"deltaLearn" will be added automatically.'
    )
    # sample_widget: Widget = MyComboBox('sampling', ['random', 'none', 'structure-based', 'farthest-point'], 
    #                                     'type S of data set sampling into splits')
    
    dataset_sample_and_num: Widget = MyLabelFrame('dataset sample and num', [
        HorizontalFrame([
            VerticalFrame([
                MyComboBox('sampling', ['random', 'none', 'structure-based', 'farthest-point'], 
                            'type S of data set sampling into splits'),
                KW_Entry('Nuse', 'N first entries of the data set file to be used'),
            ]),
            VerticalFrame([
                KW_Entry('Ntrain', r'number of the training points [0.8, i.e. 80% of the data set, by default]'),
                KW_Entry('Ntest', 'number of the test points [remainder of data set, by default]'),
            ]),
            VerticalFrame([
                KW_Entry('Nsubtrain', f'number of the sub-training points [0.8, 80% of the training set, by default]'),
                KW_Entry('Nvalidate', 'number of the validation points [remainder of the training set, by default]'), 
            ]),
        ]),
    ])
    use_ML_model_task: List[Widget] = [ 
        HorizontalFrame([
            MyLabelFrame('Input files', [
                EntryOfFileBtn('MLmodelIn', 'file with ML model', 'ML model'),
                EntryOfFileBtn('XYZfile', 'file with xyz coordinates', 'XYZ coordinates'),
                EntryOfFileBtn('XfileIn', 'file with input vectors X', 'or X (input vectors)'),   
            ], default_kw='useMLmodel', description='Use ML model to make prediction \n keyword "useMLmodel" will be added automatically'),
            deepcopy(delta_learn_widget),
        ]),
        HorizontalFrame([
            MyLabelFrame('Output files (at least one is required)', [
                EntryOfFileBtn('YestFile', 'file with estimated Y values', 'estimated Y values'), 
                EntryOfFileBtn('YgradEstFile', 'file with estimated gradients', 'estimated gradients'),
                EntryOfFileBtn('YgradXYZestFile', 'file with estimated XYZ gradients', 'estimated XYZ gradients'),
            ]),
	    Check_Button('other option', ['debug'], ['additional output for debugging']),
        ]),
    ]
    input_file_frame_widget: Widget = MyLabelFrame('Input files', [
            EntryOfFileBtn('XYZfile', 'file with XYZ coordinates', 'XYZ coordinates'),
            EntryOfFileBtn('XfileIn', 'file with input vectors X', 'or X (input vectors)'),
            EntryOfFileBtn('Yfile', 'file with reference values', 'Y (for reference)'),
            EntryOfFileBtn('YgradXYZfile', 'file with reference XYZ gradients', 'and/or gradient of XYZ')
    ])
    self_correct_widget: Widget = Check_Button('', ['selfCorrect'], ['Use self-correcting ML'], ['Use self-correcting ML'])
    create_ML_model_task: List[Widget] = [
        ML_algorithm,
        HorizontalFrame([
            MyLabelFrame('Input files', [
                EntryOfFileBtn('XYZfile', 'file with XYZ coordinates', 'XYZ coordinates'),
                EntryOfFileBtn('XfileIn', 'file with input vectors X', 'or X (input vectors)'),
                EntryOfFileBtn('Yfile', 'file with reference values', 'Y (for reference)'),
                EntryOfFileBtn('YgradXYZfile', 'file with reference XYZ gradients', 'and/or gradient of XYZ')
            ], default_kw='createMLmodel', description='Create ML model \n keyword "createMLmodel" will be added automatically'),
            deepcopy(delta_learn_widget),
        ]),
        HorizontalFrame([
            MyLabelFrame('Output files', [
                KW_Entry('MLmodelOut', 'file with ML model', 'ML model output'),
                KW_Entry('XfileOut', 'file S with X values'),
                KW_Entry('XYZsortedFileOut', 'file S with sorted XYZ coordinates\nonly works for\nmolDescrType=RE molDescrType=sorted'),
                KW_Entry('YestFile', 'file S with estimated Y values'),
                KW_Entry('YgradEstFile', 'file S with estimated gradients'),
                KW_Entry('YgradXYZestFile', 'file S with estimated XYZ gradients')
            ]),
            VerticalFrame([
            MyLabelFrame('Additional optional arguments', [
                    EntryOfFileBtn('iTrainIn', 'file with indices of training points'),
                    EntryOfFileBtn('iSubtrainIn', 'file with indices of sub-training points'),
                    EntryOfFileBtn('iValidateIn', 'file with indices of validation points'),
                    EntryOfFileBtn('iCVoptPrefIn', 'prefix of files with indices for CVopt'),
                ], default_kw='sampling=user-defined', description='"sampling=user-defined" will be added automatically.'
            ),
            deepcopy(self_correct_widget),
            ]),
        ]),
        deepcopy(dataset_sample_and_num),
    ]
    estimate_ML_accuracy_task: List[Widget] = [ deepcopy(ML_algorithm), HorizontalFrame([
        VerticalFrame([
            MyLabelFrame('Input files', [
                EntryOfFileBtn('XYZfile', 'file with XYZ coordinates', 'XYZ coordinates'),
                EntryOfFileBtn('XfileIn', 'file with input vectors X', 'or X (input vectors)'),
                EntryOfFileBtn('Yfile', 'file with reference values', 'Y (for reference)'),
                EntryOfFileBtn('YgradXYZfile', 'file with reference XYZ gradients', 'and/or gradient of XYZ')
            ], default_kw='estAccMLmodel', description='Estimate ML model \n keyword "estAccMLmodel" will be added automatically'),
            HorizontalFrame([
                MyLabelFrame('Output files', [
                    KW_Entry('MLmodelOut', 'file with ML model', 'ML model output'),
                    KW_Entry('XfileOut', 'file S with X values'),
                    KW_Entry('XYZsortedFileOut', 'file S with sorted XYZ coordinates\nonly works for\nmolDescrType=RE molDescrType=sorted'),
                    KW_Entry('YestFile', 'file S with estimated Y values'),
                    KW_Entry('YgradEstFile', 'file S with estimated gradients'),
                    KW_Entry('YgradXYZestFile', 'file S with estimated XYZ gradients')
                ]),
                VerticalFrame([deepcopy(delta_learn_widget), deepcopy(self_correct_widget)]),
            ]),
        ]),
        MyLabelFrame('Additional optional arguments', [
                EntryOfFileBtn('iTrainIn', 'file with indices of training points'),
                EntryOfFileBtn('iTestIn', 'file with indices of test points'),
                EntryOfFileBtn('iCVtestPrefIn', 'prefix of files with indices for CVtest'),
                EntryOfFileBtn('iSubtrainIn', 'file with indices of sub-training points'),
                EntryOfFileBtn('iValidateIn', 'file with indices of validation points'),
                EntryOfFileBtn('iCVoptPrefIn', 'prefix of files with indices for CVopt'),
                KW_Entry('iTrainOut', 'file S with indices of training points'),
                KW_Entry('iTestOut', 'file S with indices of test points'),
                KW_Entry('iSubtrainOut', 'file S with indices of sub-training points'),
                KW_Entry('iValidateOut', 'file S with indices of validation points'),
                KW_Entry('iCVtestPrefOut', 'prefix S of files with indices for CVtest'),
                KW_Entry('iCVoptPrefOut', 'prefix S of files with indices for CVopt'),
                EntryOfFileBtn('MLmodelIn', 'file S with ML model') # todo: wrong
            ], default_kw='sampling=user-defined', description='"sampling=user-defined" will be added automatically.'
        ),
    ]), deepcopy(dataset_sample_and_num), ]
    # learning_curve_task: List[Widget] = [
    #     ML_algorithm,
    #     HorizontalFrame([
    #         deepcopy(input_file_frame_widget),
    #         VerticalFrame([
    #             MyLabelFrame('required arguments', [
    #                 KW_Entry('lcNtrains', 'training set sizes \n example: lcNtrains=N,N,N,...,N')
    #             ], default_kw='learningCurve', description='learning curve task \n keyword "learningCurve" will be added automatically'),
    #             MyLabelFrame('optional arguments', [
    #                 KW_Entry('lcNrepeats', 'lcNrepeats=N,N,N,...,N   numbers of repeats for each Ntrain \n or:         lcNrepeats=N             number  of repeats for all  Ntrains [3 repeats default]')
    #             ]),
    #         ]),
    #     ]),
    #     MyLabel(
    #         '  Output files in directory learningCurve:\n' \
    #         '   results.json               JSON database file with all results\n' \
    #         '   lcy.csv                    CSV  database file with results for values\n' \
    #         '   lcygradxyz.csv             CSV  database file with results for XYZ gradients\n' \
    #         '   lctimetrain.csv            CSV  database file with training   timings\n' \
    #         '   lctimepredict.csv          CSV  database file with prediction timings\n' \
    #     ),
    #     # *deepcopy(create_or_est_ML_model)
    # ]
    learning_curve_task: List[Widget] = [ 
        deepcopy(ML_algorithm), 
        HorizontalFrame([
            VerticalFrame([
                MyLabelFrame('Input files', [
                    EntryOfFileBtn('XYZfile', 'file with XYZ coordinates', 'XYZ coordinates'),
                    EntryOfFileBtn('XfileIn', 'file with input vectors X', 'or X (input vectors)'),
                    EntryOfFileBtn('Yfile', 'file with reference values', 'Y (for reference)'),
                    EntryOfFileBtn('YgradXYZfile', 'file with reference XYZ gradients', 'and/or gradient of XYZ')
                ], default_kw='learningCurve', description='Estimate ML model \n keyword "learningCurve" will be added automatically'),
                HorizontalFrame([
                    MyLabelFrame('Output files', [
                        KW_Entry('MLmodelOut', 'file with ML model', 'ML model output'),
                        KW_Entry('XfileOut', 'file S with X values'),
                        KW_Entry('XYZsortedFileOut', 'file S with sorted XYZ coordinates\nonly works for\nmolDescrType=RE molDescrType=sorted'),
                        KW_Entry('YestFile', 'file S with estimated Y values'),
                        KW_Entry('YgradEstFile', 'file S with estimated gradients'),
                        KW_Entry('YgradXYZestFile', 'file S with estimated XYZ gradients')
                    ]),
                    VerticalFrame([deepcopy(delta_learn_widget), deepcopy(self_correct_widget)]),
                ]),
            ]),
            MyLabelFrame('Additional optional arguments', [
                    EntryOfFileBtn('iTrainIn', 'file with indices of training points'),
                    EntryOfFileBtn('iTestIn', 'file with indices of test points'),
                    EntryOfFileBtn('iCVtestPrefIn', 'prefix of files with indices for CVtest'),
                    EntryOfFileBtn('iSubtrainIn', 'file with indices of sub-training points'),
                    EntryOfFileBtn('iValidateIn', 'file with indices of validation points'),
                    EntryOfFileBtn('iCVoptPrefIn', 'prefix of files with indices for CVopt'),
                    KW_Entry('iTrainOut', 'file S with indices of training points'),
                    KW_Entry('iTestOut', 'file S with indices of test points'),
                    KW_Entry('iSubtrainOut', 'file S with indices of sub-training points'),
                    KW_Entry('iValidateOut', 'file S with indices of validation points'),
                    KW_Entry('iCVtestPrefOut', 'prefix S of files with indices for CVtest'),
                    KW_Entry('iCVoptPrefOut', 'prefix S of files with indices for CVopt'),
                    EntryOfFileBtn('MLmodelIn', 'file S with ML model') # todo: wrong
                ], default_kw='sampling=user-defined', description='"sampling=user-defined" will be added automatically.'
            ),
        ]), 
        deepcopy(dataset_sample_and_num),
        MyLabel(
            '  Output files in directory learningCurve:\n' \
            '   results.json               JSON database file with all results\n' \
            '   lcy.csv                    CSV  database file with results for values\n' \
            '   lcygradxyz.csv             CSV  database file with results for XYZ gradients\n' \
            '   lctimetrain.csv            CSV  database file with training   timings\n' \
            '   lctimepredict.csv          CSV  database file with prediction timings\n' \
        ),
        HorizontalFrame([
            MyLabelFrame('required arguments', [
                KW_Entry('lcNtrains', 'training set sizes \n example: lcNtrains=N,N,N,...,N')
            ], default_kw='learningCurve', description='learning curve task \n keyword "learningCurve" will be added automatically'),
            MyLabelFrame('optional arguments', [
                KW_Entry('lcNrepeats', 'lcNrepeats=N,N,N,...,N   numbers of repeats for each Ntrain \n or:         lcNrepeats=N             number  of repeats for all  Ntrains [3 repeats default]')
            ]),
        ]),
        
    ]
    cross_section_task: List[Widget] = [
        MyLabelFrame('cross section option', [
            KW_Entry('nExcitations', 'number of excited states to calculate. (default=3)'),
            KW_Entry('nQMpoints', 'user-defined number of QM calculations for training ML. (default=0, number of QM calculations will be determined iteratively)'),
            KW_Entry('deltaQCNEA', 'define the broadening parameter of QC-NEA cross section'),
            KW_Entry('nMaxPoints', 'maximum number of QC calculations in the iterative procedure. (default=10000)'),
            KW_Entry('nNEpoints', 'number of ML ensemble prediction. (default=50000)'),
            Check_Button('', ['plotQCNEA', 'plotQCSPC'], 
                        ['requests plotting QC-NEA cross section', 'requests plotting cross section obtained via single point convolution'])
        ], default_kw='crossSection', description='using ML-NEA method to simulate absorption spectrum ')
    ]
    geomopt_task: List[Widget] = [
        MyLabelFrame('opt option', [
            MyComboBox('optprog', ['Gaussian', 'ASE'], 'Geometry optimization program'),
            MyComboBox('opttask', ['opt', 'ts', 'irc', 'freq'], 'Geometry optimization task S (ASE program can only use with task=opt)\n' \
                                                                'opt [default]        Optimization for energy minimum\n' \
                                                                'ts                   Optimization for transition state\n' \
                                                                'freq                 Frequence analysis for input geometry\n',),
        ], default_kw='geomopt'),
        EntryOfFileBtn('xyzfile', 'file with initial XYZ geometry'),
        MyComboBox('MLmodelType', ['KREG', 'ANI1x', 'ANI2x', 'ANI1ccx'], 'choose the ML model'),
        MyLabelFrame('choose KREG model file', [
            EntryOfFileBtn('MLmodelIn', 'choose your KREG model file'),
        ]),
        MyLabelFrame('ASE option', [
            KW_Entry('ase.fmax', 'threshold of maximum force (in eV/A)\n[default values: 0.0005]'),
            KW_Entry('ase.steps', 'maximum steps\n[default values: 1000]'),
            MyComboBox('ase.optimizer', ['LBFGS', 'BFGS'], 'optimizer, default is LBFGS')
        ], description=' these options are only used for ASE program')
    ]

    ##### Dataset tasks #####
    xyz2x_task: List[Widget] = [
        MyLabelFrame('required arguments', [
            EntryOfFileBtn('MLmodelIn', 'file with ML model'),
            EntryOfFileBtn('XYZfile', 'file with XYZ coordinates'),
            EntryOfFileBtn('XfileOut', 'file with X values')
        ], default_kw='XYZ2X', description='keyword "XYZ2X" will be added automatically'),
        MyLabelFrame('Optional arguments specifying descriptor', [
            MyComboBox('molDescriptor', ['RE', 'CM'],
                        '      RE [default]             vector {Req/R}, where R is internuclear distance\n'
                        '      CM                       Coulomb matrix'
            ),
            MyComboBox('molDescrType', ['sorted', 'unsorted', 'permuted'],
                '      sorted                   default for molDescrType=CM\n'
                '                               sort by:\n'
                '                                 norms of CM matrix (for molDescrType=CM)\n'
                '                                 nuclear repulsions (for molDescrType=RE)\n'
                '      unsorted                 default for molDescrType=RE\n'
                '      permuted                 molecular descriptor with all atom permutations'
            ),
            MyLabelFrame('additional option for molDescrType=sorted and molDescriptor=RE', [
                KW_Entry('XYZsortedFileOut', 'file with sorted XYZ coordinates')
            ]),
            MyLabelFrame('additional option for molDescrType=permuted', [
                KW_Entry('permInvGroups', 
                        'permutationally invariant groups S\n'
                        'e.g. for water dimer permInvGroups=1,2,3-4,5,6\n'
                        'permute water molecules (atoms 1,2,3 and 5,6,7)'
                ),
                KW_Entry('permInvNuclei', 
                        'permutationally invariant nuclei S\n'
                        'e.g.permInvNuclei=2-3.5-6\n'
                        'will permute atoms 2,3 and 6,7'
                )
            ])
        ]),
    ]
    analyze_task: List[Widget] = [
        MyLabelFrame('For reference data', [
            EntryOfFileBtn('Yfile', 'file S with values'),
            EntryOfFileBtn('Ygrad', 'file S with gradients'),
            EntryOfFileBtn('YgradXYZfile', 'file S with gradients in XYZ coordinates')
        ], default_kw='analyze', description='keyword analyze will be added automatically'),
        MyLabelFrame('for estimated data', [
            EntryOfFileBtn('YestFile', 'file S with estimated Y values'),
            EntryOfFileBtn('YgradEstFile', 'file S with estimated gradients'),
            EntryOfFileBtn('YgradXYZestFile', 'file S with estimated XYZ gradients'),
        ])
    ]
    sample_task: List[Widget] = [
        HorizontalFrame([
            MyLabelFrame('required arguments', [
                MyLabelFrame('data set', [
                    EntryOfFileBtn('XYZfile', 'file S with xyz coordinates'),
                    EntryOfFileBtn('XfileIn', 'file S with input vectors X'), 
                ]),
                MyLabelFrame('splitted data index file', [
                    KW_Entry('iTrainOut', 'file S with indices of training points'),
                    KW_Entry('iTestOut', 'file S with indices of test points'),
                    KW_Entry('iSubtrainOut', 'file S with indices of sub-training points'),
                    KW_Entry('iValidateOut', 'file S with indices of validation points'),
                ]),
                MyLabelFrame('Cross-validation', [
                    MyLabelFrame('CVtest', [
                        KW_Entry('NcvTestFolds', 'sets number of folds to N [5 by default]'),
                        Check_Button('', ['LOOtest'], ['leave-one-out cross-validation']),
                        KW_Entry('iCVtestPrefOut', 'prefix S of files with indices for CVtes'),
                    ], default_kw='CVtest', description='keyword "CVtest" will be added automatically'),
                    MyLabelFrame('CVopt', [
                        KW_Entry('CVopt', 'sets number of folds to N [5 by default]'),
                        Check_Button('', ['LOOopt'], ['leave-one-out cross-validation']),
                        KW_Entry('iCVoptPrefOut', 'prefix S of files with indices for CVopt'),
                    ], default_kw='CVopt', description='keyword "CVopt" will be added automatically'),
                ])
            ], default_kw='sample', description='keyword "sample" will be added automatically'),
            MyLabelFrame('optional arguments', [
                MyComboBox('sampling', ['random', 'none', 'structure-based', 'farthest-point'], 
                            'type S of data set sampling into splits'),
                KW_Entry('Nuse', 'N first entries of the data set file to be used'),
                KW_Entry('Ntrain', r'number of the training points [0.8, i.e. 80% of the data set, by default]'),
                KW_Entry('Ntest', 'number of the test points [remainder of data set, by default]'),
                KW_Entry('Nsubtrain', f'number of the sub-training points [0.8, 80% of the training set, by default]'),
                KW_Entry('Nvalidate', 'number of the validation points [remainder of the training set, by default]'), 
            ]),
        ]),
    ]
    slice_task: List[Widget] = [
        MyLabelFrame('required argument', [
            EntryOfFileBtn('XfileIn', 'file S with input vectors X'),
            EntryOfFileBtn('eqXfileIn', 'file S with input vector for equilibrium geometry')
        ], default_kw='slice', description='keyword "slice" will be added automatically'),
        MyLabelFrame('optional argument', [
            KW_Entry('Nslices', 'number of slices [3 by default]'),
        ]),
    ]
    samp_from_slice_task: List[Widget] = [
        MyLabelFrame('required argument', [
            KW_Entry('Ntrain', 'total integer number N of training points from all slices'),
        ], default_kw='sampleFromSlices', description='keyword "sampleFromSlices" will be added automatically'),
        MyLabelFrame('optional argument', [
            KW_Entry('Nslices', 'number of slices [3 by default]'),
            MyComboBox('sampling', ['random', 'none', 'structure-based', 'farthest-point'], 
                                        'type S of data set sampling into splits')
            # deepcopy(sample_widget),
        ])
    ]
    merge_slice_task: List[Widget] = [
        MyLabelFrame('required argument', [
            KW_Entry('Ntrain', 'total integer number N of training points from all slices'),
        ], default_kw='mergeSlices', description='keyword "mergeSlices" will be added automatically'),
        MyLabelFrame('optional argument', [
            KW_Entry('Nslices', 'number of slices [3 by default]')
        ])
    ]
    
    ##### main structure #####
    ML_task: List[Widget] = [ 
        MyNotebook({
            'use ML model': use_ML_model_task + [OutputBtn([use_ML_model_task])],
            'create ML model': create_ML_model_task + [OutputBtn([create_ML_model_task])],
            'estimate accuracy': estimate_ML_accuracy_task + [OutputBtn([estimate_ML_accuracy_task])],
            'learning curve': learning_curve_task + [OutputBtn([ learning_curve_task])],
            'spectrum simulation': cross_section_task + [OutputBtn(cross_section_task)],
            # 'geomopt': geomopt_task + [OutputBtn(geomopt_task)],
        }),
    ]
    data_set_task: List[Widget] = [
        MyNotebook({
            'convert XYZ to X': xyz2x_task + [OutputBtn(xyz2x_task)],
            'analyze': analyze_task + [OutputBtn(analyze_task)],
            'sample': sample_task + [OutputBtn(sample_task)],
            'slice': slice_task + [OutputBtn(slice_task)],
            'sample from slice': samp_from_slice_task + [OutputBtn(samp_from_slice_task)],
            'merge slice':merge_slice_task + [OutputBtn(merge_slice_task)],
        })
    ]

    ##### load all widget #####
    gui = GUI()
    gui.title('MLatom GUI')
    gui.add_item(gui, [
            MyNotebook({
                'ML tasks': ML_task,
                'Data set tasks': data_set_task,
            }),
        ]
    )
    gui.update()
    gui.deiconify()
    gui.mainloop()

if __name__ == '__main__':
    main()
