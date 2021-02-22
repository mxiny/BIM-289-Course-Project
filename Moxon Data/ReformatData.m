%   BY: Norma Gowans
%
%   CREATED: 20210222
%
%   PURPOSE:
%              - reformat data from dr. moxon
%
%   IMPORTANT VARIABLES:
% 
%             filename_meta - contains specific information about the
%             monkey and data collection
%             
%             selected_data.LEFT - table of the collected and viable data for 14
%             neurons 
%
%             psth_struct.LEFT.event_11.sig002a.relative_response -
%             spiking data for one neuron with 100 trials and length 150
%             bins
%

clear
clc
close all

%% Import Original Data

% folder path
dirName = 'C:\Users\14087\Desktop\BIM Project\Moxon Data\';  

% list all files with mat extension
files = dir( fullfile(dirName,'*.mat') );   

% list the mat file names
files = {files.name}';                 

% load the files
load(files{1})

%% Selected Data

% the data is in a table
T = selected_data.LEFT;
allNeuronData = T.('channel_data');
neuronData = allNeuronData{1};
size(neuronData)

%% Plot PSTH

% viewing the psth provided
bar(psth_struct.LEFT.event_11.sig002a.psth)




