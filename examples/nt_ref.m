function nt_ref(name)
% Run the NoiseTools routines on examples/<name>/data.mat, save nt_out.mat beside it.
% cd to this folder first, then:  nt_ref synthetic   /   nt_ref eeg

nt = getenv('NOISETOOLS');
if isempty(nt)
    nt = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'NoiseTools');
end
if ~exist(nt, 'dir')
    error('NoiseTools not found at %s -- set the NOISETOOLS environment variable', nt);
end
addpath(nt);

here = fullfile(fileparts(mfilename('fullpath')), name);
f = fullfile(here, 'data.mat');
if ~exist(f, 'file')
    error('%s is missing -- run "python %s/gen.py" first', f, name);
end
C = load(f);

nlags = double(C.p_nlags);
widths = double(C.p_widths(:))';
smoothT = double(C.p_smoothT);
dorder = double(C.p_detrend_order);
sthresh = double(C.p_star_thresh);
cthresh = double(C.p_cca_thresh);
shifts = double(C.p_cca_shifts(:))';

out = struct();
for i = 1:numel(C.case_names)
    nm = C.case_names{i};
    X = C.([nm '_X']);  Y = C.([nm '_Y']);
    keep = double(C.([nm '_keep']));
    [n, c] = size(X);
    fprintf('%-9s %5dx%-4d ', nm, n, c); t0 = tic;

    Xdm = nt_demean(X);  Ydm = nt_demean(Y);
    out.([nm '__demean']) = Xdm;
    out.([nm '__normcol']) = nt_normcol(X);
    out.([nm '__cov_raw']) = nt_cov(X);
    out.([nm '__cov_dm']) = nt_cov(Xdm);

    [V, ev] = nt_pcarot(nt_cov(Xdm));
    out.([nm '__pcarot_V']) = V;
    out.([nm '__pcarot_ev']) = ev(:);
    out.([nm '__pca']) = nt_pca(Xdm, 0, keep);

    if n > nlags
        out.([nm '__multishift']) = nt_multishift(X, 0:nlags-1);
    end
    if n > max(widths)
        out.([nm '__smooth']) = nt_smooth(X, smoothT);
        out.([nm '__multismooth']) = nt_multismooth(X, widths);
    end

    % real EEG already drifts and glitches, so those cases reuse X
    if isfield(C, [nm '_Xd']); Xd = C.([nm '_Xd']); else; Xd = X; end
    out.([nm '__detrend']) = nt_detrend(Xd, dorder);

    if isfield(C, [nm '_Xg']); Xg = C.([nm '_Xg']); else; Xg = X; end
    if c >= 4
        [ys, ws] = nt_star(Xg, sthresh);
        out.([nm '__star']) = ys;
        out.([nm '__star_w']) = ws;
    end

    j = nt_cov([Ydm, Xdm]);  q = size(Y, 2);
    r = nt_regcov(j(1:q, q+1:end), j(q+1:end, q+1:end));
    out.([nm '__regcov']) = r;
    out.([nm '__regcov_pred']) = Xdm * r;

    [A, B, R] = nt_cca(Xdm, Ydm, [], [], [], cthresh);
    out.([nm '__cca_A']) = A;
    out.([nm '__cca_B']) = B;
    out.([nm '__cca_R']) = R(:);

    if n > 2*max(abs(shifts)) + 10
        [~, ~, Rs] = nt_cca(Xdm, Ydm, shifts, [], [], cthresh);
        out.([nm '__cca_shiftR']) = Rs;
    end
    fprintf('%.1fs\n', toc(t0));
end

out.nt_version = nt_version;
out.matlab_ver = version;
save(fullfile(here, 'nt_out.mat'), '-struct', 'out', '-v7');
fprintf('wrote %s\n', fullfile(here, 'nt_out.mat'));
