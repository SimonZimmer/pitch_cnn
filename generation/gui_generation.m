function varargout = gui_generation(varargin)
% GUI_GENERATION MATLAB code for gui_generation.fig

% Begin initialization code
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_generation_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_generation_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code


% --- Executes just before gui_generation is made visible.
function gui_generation_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
handles.output = hObject;
set(handles.pushbutton1,'String','generate');
set(handles.pushbutton2,'String','play sound');
menu_items = {"linear", "exponential", "hyperbolic", "random", "lin_reciprocal", "exp_reciprocal"};
set(handles.popupmenu1,'string', menu_items);
handles.current_selection = "linear";

duration = 1.0;
Ts = 1/(2^14);
handles.t = 0 : Ts : duration-Ts;
handles.freq_value = 0;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes gui_generation wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = gui_generation_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit1_Callback(hObject, eventdata, handles)
freq_value = str2double(get(handles.edit1, 'String'));
handles.freq_value = freq_value;
guidata(hObject,handles)

% --- Executes on selection change in edit2.
function edit2_Callback(hObject, eventdata, handles)
root_gain = str2double(get(handles.edit2, 'String'));
handles.root_gain = root_gain;
guidata(hObject,handles)

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
handles.root = root_note(handles.freq_value, 2^14, 1.0);
guidata(hObject,handles)
handles.harmonic_tones = harmonics(handles.current_selection, 10, handles.freq_value, 2^14, handles.t, 0);
guidata(hObject,handles)
handles.output_sum = handles.root_gain * handles.root +  sum(handles.harmonic_tones, 2);
guidata(hObject,handles)
drawplot(handles)

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
sound(handles.output_sum);

% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
contents = cellstr(get(handles.popupmenu1,'String'));
handles.current_selection = contents{get(handles.popupmenu1, 'Value')};
guidata(hObject,handles);

function drawplot(handles)

axes(handles.axes1);
plot(handles.t, handles.output_sum, 'b')
xlim([2*1/handles.freq_value,12*1/handles.freq_value])
title('time domain')
xlabel('time in s')
ylabel('level')

max_ref = max(abs(fft(handles.root(:,1))));

axes(handles.axes2);
cla reset 
x_magnitude = abs(fft(handles.root(:,1)));
x_magnitude  = x_magnitude(1:end/2+1,:)';
x_magnitude = handles.root_gain * x_magnitude / max_ref;
plot(x_magnitude,'b')
hold on;
x_magnitude = abs(fft(sum(handles.harmonic_tones, 2)));
x_magnitude  = x_magnitude(1:end/2+1,:)';
x_magnitude = x_magnitude / max_ref;
plot(x_magnitude,'r')
hold off
legend('root', 'harmonics')
title('frequency domain')
xlabel('frequency in Hz')
ylabel('level')
ylim([0,1])
