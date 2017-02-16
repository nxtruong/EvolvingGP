classdef SignalsModel < handle
    % Class to manage the MISO signals in and out of a GP model.
    
    properties(GetAccess=public,SetAccess=public)
        time;
        m_k;
        m_maxk;
        m_maxlag;
        m_Nregressors;
        
        m_I = {};         % names of the input signals
        m_Ilags = {};     % lag vectors of the input signals
        m_O = '';         % name of the output signal
        Signals;          % the values of the signals, as fields of this structure
        m_Ninputs;
        m_Noutputs;
        m_mean;           % mean of signals
        m_std;            % std of signals
        normalize;      % whether normalized or not
    end
        
    methods
        function self = SignalsModel(N)
            % Construct the SignalsModel object.
            %   SignalsModel(N)
            % Inputs:
            %   N   - Maximum number of data points.
            
            self.m_maxk = N;
            self.time = NaN(N,1);
            self.m_k = 0;
            self.normalize = false;
            self.m_mean = struct;
            self.m_std = struct;
        end
        
        function setValue(self, signal, value, k)
            % Set the value of a signal at time/index k.
            % Do not update m_k nor the statistical moments.
            assert(k <= self.m_maxk);
            self.Signals.(signal)(k) = value;
        end
        
        function appendValues(self, varargin)
            % Advance m_k by one step and append the values of the signals,
            % unless m_k is already >= m_maxk.
            % The values can be entered in two ways:
            % - By a structure whose fields are signal names.
            % - By pairs of 'signalname', value.
            % Signals that are not set will keep the current value
            % (default: NaN).
            
            if self.m_k >= self.m_maxk
                warning('Current time step is already at the maximum size.');
                return;
            end
            self.m_k = self.m_k + 1;
            
            if nargin == 1, return; end
            
            if length(varargin) == 1
                assert(isstruct(varargin{1}));
                values = varargin{1};
            elseif mod(length(varargin), 2) == 0
                values = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);
            else
                error('Invalid arguments.');
            end
            
            flds = fieldnames(values);
            for ii = 1:numel(flds)
                fld = flds{ii};
                if isfield(self.Signals, fld)
                    self.Signals.(fld)(self.m_k) = values.(fld);
                end
            end
            
            if self.normalize
                self.updateStatMoments();
            end
        end
        
        function x = getRegressorVectors(self, k, varargin)
            % Returns the regressor vectors as rows of the output matrix.
            %       X = getRegressorVectors(self, k, opt)
            % Inputs:
            %   k   - the index until which the regressor vectors are
            %           generated; or the time-step vector.
            %   opt - 'force' if time steps in k that are <= maxlag will be
            %           removed; otherwise, it's assumed that all time
            %           steps in k are > maxlag
            
            if nargin==1, k = self.m_k; end;
            opt = 0;
            if (nargin==3 && strcmpi(varargin{1},'force')), opt=1; end
            assert(~isempty(k), 'SignalsModel:getRegressorVectors','Empty time-step vector k');
            
            if opt == 0
                assert(min(k) > self.m_maxlag,'SignalsModel:InvalidSignalIndex','Too small time-instant. The regressor vector requires lagged values time-instant k=>0');
            else
                k(k <= self.m_maxlag) = [];
            end
            
            k = k(:);
            n = length(k);
            x = NaN(n, self.m_Nregressors);
            j = 1;
            inputs_without_regressors = cellfun(@isempty,self.m_Ilags);
            for i = 1:self.m_Ninputs
                if inputs_without_regressors(i),
                    continue;
                end
                
                nr = numel(self.m_Ilags{i});
                x(:,j:j+nr-1) = self.Signals.(self.m_I{i})(bsxfun(@minus,k,self.m_Ilags{i}));
                if self.normalize
                    x(:,j:j+nr-1) = (x(:,j:j+nr-1) - self.m_mean.(self.m_I{i}))./self.m_std.(self.m_I{i});
                end
                j = j + nr;
            end
            if any(any(isnan(x))) && opt==0
                error('SignalsModel:InvalidSignalIndex','Invalid regressor values.');
            end
        end
        
        function t = getRegressand(self, k)
            % Returns the regressand / output vector at time steps in k
            if nargin == 1, k = self.m_k; end
            k = k(:);
            
            t = self.Signals.(self.m_O)(k);
            if self.normalize
                t = (t-self.m_mean.(self.m_O))./self.m_std.(self.m_O);
            end
            if any(isnan(t))
                error('SignalsModel:InvalidSignalIndex','Invalid regressor values.');
            end
        end
        
        function setInputs(self, inputs, signalnames)
            %	setInputs(self, (struct) inputs, (cell) orders)
            %
            % Set the input signals for modelling.
            % Inputs:
            %   inputs  - a struct containing attributes named as the
            %               signals names. The value of each (attribute)
            %               name contains a vector that desribes which
            %               delayed signal values (l1,..,ln) from current
            %               time-step k are used for building a regressor
            %               vector.
            %   orders  - an optional cell array that specifies the order
            %               of the input signals (in inputs) in the
            %               regressor vector; if not provided, the order of
            %               the field names in inputs structure is used.
            %
            % example:
            %	inputs.u=[1,3,5];
            %	inputs.y=[1,2];
            %	%in this case the feature vector (of current time-step k) for prediction is: x(k)=[u(k-1) u(k-3) u(k-5) y(k-1) y(k-2)];
            %	"obj".setInputs(inputs);
            if nargin > 2
                assert(iscellstr(signalnames), 'Argument ORDERS must be a cell array of string.');
                assert(all(all(isfield(inputs, signalnames))), 'Argument ORDERS must contain only field names in INPUTS.');
            else
                signalnames = fieldnames(inputs);
            end
            Nin = length(signalnames);
            
            self.m_I = cell(1, Nin);
            self.m_Ilags = cell(1, Nin);
            for i = 1:Nin
                self.Signals.(signalnames{i}) = NaN(self.m_maxk,1);
                self.m_I{i} = signalnames{i};
                self.m_Ilags{i} = inputs.(signalnames{i});
                self.m_mean.(signalnames{i}) = NaN;
                self.m_std.(signalnames{i}) = NaN;
            end
            self.m_Ninputs = Nin;
            self.m_maxlag = max([self.m_Ilags{:}]);
            self.m_Nregressors = numel([self.m_Ilags{:}]);
        end
        
        function setOutput(self, output)
            %	setOutput(self,(str) output)
            %
            % set the output (target) signal for modelling
            if ~isfield(self.Signals, output)
                self.Signals.(output) = NaN(self.m_maxk, 1);
                self.m_mean.(output) = NaN;
                self.m_std.(output) = NaN;
            end
            self.m_O = output;
            self.m_Noutputs = 1;
        end
        
        function [regressorVector_mask, regressand_mask, x, o] = filterUndefinedDataAt(self,k,varargin)
            % Filter undefined (NaN) data at given time steps in K.
            regressorVector_mask = [];
            regressand_mask = [];
            x = [];
            o = [];
            
            if nargin==3
                warntype = varargin{1}; % warning, error or quiet
            else
                warntype = 'warning';
            end
            k = k(:);
            
            shortk = (k<=self.m_maxlag);
            if any(shortk)
                self.(warntype)('SignalsModel:InvalidSignalIndex','Too small time-step. The regressor vector requires lagged values at/before time-step k=0');
            end
            
            k1 = k(~shortk);
            if isempty(k1)
                return;
            end
            
            n = length(k1);
            x = NaN(n,self.m_Nregressors);
            j = 1;
            inputs_without_regressors = cellfun(@isempty,self.m_Ilags);
            for i = 1:self.m_Ninputs
                if (inputs_without_regressors(i)==1), continue; end
                nr = numel(self.m_Ilags{i});
                x(:,j:j+nr-1) = self.Signals.(self.m_I{i})(bsxfun(@minus,k1,self.m_Ilags{i}));
                j = j + nr;
            end
            
            o = self.Signals.(self.m_O)(k1);
            xsum = sum(x,2);
            nanx = isnan(xsum);
            nano = isnan(o);
            if any(nanx)>0 || any(nano)>0
                self.(warntype)('SignalsModel:InvalidSignalIndex','Invalid regressor values.');
            end
            
            k_mask = false(size(k));
            regressorVector_mask = k_mask;
            regressand_mask = k_mask;
            
            regressorVector_mask(~shortk) = ~nanx;
            regressand_mask(~shortk) = ~nano;
        end
        
        function updateStatMoments(self)
            % select only signal segments without NaN values
            for i = 1:self.m_Ninputs
                notNaNs = ~isnan(self.Signals.(self.m_I{i}));
                if sum(notNaNs) > 1
                    self.m_mean.(self.m_I{i}) = mean(self.Signals.(self.m_I{i})(notNaNs));
                    self.m_std.(self.m_I{i}) = std(self.Signals.(self.m_I{i})(notNaNs));
                end
            end
        end
        
        function new = copy(self)
            % Copy super_prop
            new = feval(class(self), self.m_maxk);
            % Copy sub_prop1 in subclass
            % Assignment can introduce side effects
            new.setInputs(cell2struct(self.m_Ilags,self.m_I,2));
            new.setOutput(self.m_O);
            
            new.Signals = self.Signals; % copy signal data

            new.time = self.time;
            new.m_k = self.m_k;
            new.m_mean = self.m_mean;
            new.m_std = self.m_std;
            new.normalize = self.normalize;
        end
        
        function quiet(self,id,msg)
            %empty function: is a quiet function, it isn't?
        end
        
        function warning(self,id,msg)
            warning(id,msg);
        end
        
        function error(self,id,msg)
            error(id,msg);
        end
        %
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
