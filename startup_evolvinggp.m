% OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

mydir = fileparts(mfilename('fullpath'));   % what is the current folder
% if OCT && numel(mydir)==2 
%   if strcmp(mydir,'./'), mydir = [pwd,mydir(2:end)]; end
% end                 % OCTAVE 3.0.x relative, MATLAB and newer have absolute path

addpath(mydir);
clear mydir
