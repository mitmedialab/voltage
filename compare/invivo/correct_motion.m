function correct_motion(output_dir)
%%
addpath(genpath(fullfile(cd,'invivo-imaging','lib')));

%%
if exist(fullfile(output_dir,'reg_shifts.mat'),'file')
    load(fullfile(output_dir,'reg_shifts.mat'));
    xShifts = reg_shifts(1,71:end);
    yShifts = reg_shifts(2,71:end);
    dX = xShifts - mean(xShifts);
    dY = yShifts - mean(yShifts);
    dXhp = dX - smooth(dX, 2000)';  % high pass filter
    dYhp = dY - smooth(dY, 2000)';
    dXs = smooth(dXhp, 5)';  % low-pass filter just to remove some jitter in the tracking.  Not sure if necessary
    dYs = smooth(dYhp, 5)';

    tic;
    mov = shiftdim(loadtiff(fullfile(output_dir,'denoised.tif')),2);
    [ySize, xSize, nFrames] = size(mov);
    t = 1:nFrames;
    
    avgImg = mean(mov,3);
    dmov = mov - avgImg;
    
    dT = 5000;
    % First column is the start of each epoch, second column is the end
    if dT ~= nFrames; bdry = [(1:dT:nFrames)', [(dT:dT:nFrames) nFrames]'];
    else; bdry = [(1:dT:nFrames)', [nFrames]']; end;
    nepoch = size(bdry, 1);
    out4 = zeros(size(mov));
    for j = 1:nepoch;
        tau = bdry(j,1):bdry(j,2);
        [out4(:,:,tau), ~] = SeeResiduals(dmov(:,:,tau), [dXs(tau); dYs(tau); dXs(tau).^2; dYs(tau).^2; dXs(tau) .* dYs(tau)], 1);
    end;
    
    options.overwrite = true;
    saveastiff(single(out4),fullfile(output_dir,'motion_corrected.tif'),options);
    toc;
end
