% MATLAB gate for issue #159 (run after gate_prep.py, before gate_check.py).
% Loads the genuine single-model Fortran fixture and the pamica-written 2-model
% output with EEGLAB's real loadmodout15.m, and saves W/A/sbeta/rho so the Python
% companion can check they match what Python loadmodout read. This is the only
% test that pins direction (2) of the interop contract: EEGLAB reads pamica's
% (and Fortran's) bytes correctly. A pamica write->read round trip cannot.

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');
sample = fullfile(root, 'pamica', 'sample_data');

% addpath sample_data LAST so its working loadmodout15.m shadows any broken
% copy elsewhere on the path (postAmicaUtility's has a syntax error on R2025b).
addpath(sample);

dirs = struct('tag', {'fixture', 'two_model'}, ...
              'path', {fullfile(sample, 'amicaout'), fullfile(here, 'gate_2model')});

for k = 1:numel(dirs)
    % loadmodout15 concatenates paths directly, so the dir needs a TRAILING SLASH.
    d = [dirs(k).path filesep];
    m = loadmodout15(d);
    W = m.W; A = m.A; sbeta = m.sbeta; rho = m.rho; num_models = m.num_models; %#ok<NASGU>
    save(fullfile(here, ['mat_' dirs(k).tag '.mat']), 'W', 'A', 'sbeta', 'rho', 'num_models');
    fprintf('%s: num_models=%d  W=%s  A=%s\n', dirs(k).tag, m.num_models, ...
            mat2str(size(m.W)), mat2str(size(m.A)));
end

fprintf('\nSaved mat_fixture.mat and mat_two_model.mat. Next: gate_check.py.\n');
