function run_normcorre(input_file, output_dir)
%%
addpath(genpath(fullfile(cd,'invivo-imaging','lib')));

%% NoRMCorre image registration
mov=loadtiff(input_file);
[nrows, ncols, nframes] = size(mov);
movReg=NoRMCorre2(mov,output_dir); % get registered movie
clear mov
options.overwrite = true;
saveastiff(movReg,fullfile(output_dir,'movReg.tif'),options); % save registered movie
clear movReg

% extract motion traces into MAT file
reg_shifts = returnShifts(output_dir);
save(fullfile(output_dir,'reg_shifts.mat'),'reg_shifts');
