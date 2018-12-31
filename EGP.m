classdef EGP < nextgp.GP
    % Class for the evolving GP, subclassing nextgp.GP
    % Requires nextgp (aka. simplegp)
    
    properties(GetAccess=public,SetAccess=protected)
        BVtst	= [];   % timestamps / indices of the training data
        
        % Currently we only support Gaussian likelihood and a limited set
        % of covariance functions, etc., to be used in online control with
        % CasADi (as supported by nextgp). So some of the fields in the
        % original code are not used, hence commented out below.
        
        % inference method
        gpml_inf;
        gpml_lik;
        gpml_hyp;
        gpml_mean;
        gpml_cov;
        
        m_size = 0;
        
        signals;  % the SignalsModel object
    end
    
    properties (Access=public)
        % Configuration for (re)optimizing hyperparameters
        hypOptim = struct('enable', true, 'iter', 10, 'optimizer', @minimize_minfunc);
        
        % Forgetting factor
        forgetting = struct('factor', 1, 'type', 'none');
        
        % Configuration of reducing the training data set
        reducing = struct('maxSize', NaN, 'type', 'windowing'); %, 'indexLifes', NaN);
        
    end
    
    methods
        function self = EGP(signals, inf, hyp, meanf, cov, lik, X, Y)
            % Construct the EvolvingGP object with given signals and from
            % GPML-style parameters.
            % To initialize the training dataset of the GP model:
            %   - Give X (training inputs in rows) and Y (training
            %     targets); or
            %   - Give only X = K (and Y must be omitted), where K contains
            %     the indices / timestamps in signals that are used to
            %     construct the initial training set.
            
            assert(isa(signals, 'SignalsModel'), 'signals must be a SignalsModel object.');
            
            if ~exist('Y', 'var') || isempty(Y)
                % X is K = vector of indices
                k = X(:);
                [RVmask, RSmask] = signals.filterUndefinedDataAt(k, 'warning');
                k = k(RVmask & RSmask);
                assert(~isempty(k), 'Initial training data must not be empty.');
            
                % New inputs and targets
                X = signals.getRegressorVectors(k);
                Y = signals.getRegressand(k);
                
                N = length(k);
            else
                N = size(X,1);
                assert(isvector(Y) && N == size(Y,1), 'Invalid initial training dataset X,Y.');
                
                % The timestamps for all training points are 0
                k = zeros(N,1);
            end
            
            self = self@nextgp.GP(nextgp.GPData(hyp, meanf, cov, lik, X, Y));
            
            self.signals = signals;
            % self.reducing.indexLifes = NaN(signals.m_maxk,2);
            
            self.BVtst = k;
            self.m_size = N;
            
            %{
            for ii = 1:length(k)
                self.reducing.indexLifes(k(ii),:) = [signals.time(k(ii)),Inf];
            end
            %}
            
            if ~iscell(inf), inf = {inf}; end
            
            % Save the GPML parameters
            if ~iscell(inf), inf = {inf}; end
            self.gpml_inf = inf;
            
            self.gpml_hyp = hyp;
            
            if ~iscell(meanf), meanf = {meanf}; end
            self.gpml_mean = meanf;
            
            if ~iscell(cov), cov = {cov}; end
            self.gpml_cov = cov;
            
            self.gpml_lik = lik;
        end
        
        %{
        This method is disabled because the initial training dataset may
        not contain points from the stored signals that can be
        reconstructed.
        
        function resetActiveSet(self)
            k = self.BVtst;
            self.gpdata.training_data.x = [];
            self.gpdata.training_data.y = [];
            self.BVtst = [];
            
            self.include(k);
        end
        %}
        
        function resetPrior(self,default)
            if nargin<2, default = 1; end
            
            % This may not work correctly because the signals model may not contain all necessary regressors of the GP
            % D = self.signals.m_Nregressors;
            
            D = self.gpdata.input_dim;
            
            if nargin==2 && isstruct(default)
                self.gpml_hyp = default;
            else
                %initialize default values:
                self.gpml_hyp = struct();
                self.gpml_hyp.cov= ones(eval(feval(self.gpml_cov{:})),1).*default;
                self.gpml_hyp.mean=ones(eval(feval(self.gpml_mean{:})),1).*default;
                self.gpml_hyp.lik= ones(eval(feval(self.gpml_lik{:})),1).*default;
            end
            
            % Set the hyperparameters in GPData
            if isfield(self.gpml_hyp, 'cov')
                self.gpdata.cov.sethyp(self.gpml_hyp.cov, D);
            end
            if isfield(self.gpml_hyp, 'mean')
                self.gpdata.m_mean.sethyp(self.gpml_hyp.mean, D);
            end
            if isfield(self.gpml_hyp, 'lik')
                self.gpdata.m_hyplik = self.gpml_hyp.lik;
            end
            
            self.gpdata.updatePosterior();
        end
        
        function include(self, x, t, k)
            % Append new data points (X,T), where X are rows of inputs and
            % T are corresponding targets, to the training data set.
            % If T is omitted, X contains indices in the signals model for
            % the new data points.
            %
            % When both (X,T) are given, an optional argument k can be
            % given after T, which contains the indices (or timestamps) of
            % the new data points. Otherwise, the current step m_k of the
            % signal model will be used.
            
            if nargin < 3 || isempty(t)
                k = x(:);
            
                [RVmask, RSmask] = self.signals.filterUndefinedDataAt(k, 'warning');
                k = k(RVmask & RSmask);
                if isempty(k), return; end
            
                % New inputs and targets
                x = self.signals.getRegressorVectors(k);
                t = self.signals.getRegressand(k);
                
                N = length(k);
            else
                N = size(x, 1);
                t = t(:);
                
                if nargin >= 4
                    k = k(:);
                end
            end
            self.gpdata.training_data.x(end+1:end+N,:) = x;
            self.gpdata.training_data.y(end+1:end+N,:) = t;
            self.gpdata.updatePosterior();  % because we change the training dataset
            
            self.BVtst(end+1:end+N, :) = k;
            self.m_size = size(self.BVtst, 1);
            
            %{
            for ii = 1:length(k)
                self.reducing.indexLifes(k(ii),:) = [self.signals.time(k(ii)),Inf];
            end
            %}
        end
        
        function [ymu,ys2] = predictAt(self, k, varargin)
            % Predict at given time steps (k).
            % Optional arguments:
            %   extras - a matrix whose rows are additional inputs to be
            %               appended to the regressor vectors generated
            %               from the stored signals.
            %   the rest will be passed as-is to predict().
            
            k = k(:);
            ymu = NaN(length(k),1);
            ys2 = NaN(length(k),1);
            %         try
            RVmask = self.signals.filterUndefinedDataAt(k,'warning');
            newk = k(RVmask);
            %             ymu(~mask_k)=NaN;
            %             ys2(~mask_k)=NaN;
            if ~isempty(newk)
                xs = self.signals.getRegressorVectors(newk);
                % 		catch err
                % 			if strcmp(err.identifier,'SignalsModel:InvalidSignalIndex')
                % 				ymu=inf(size(k));
                % 				ys2=inf(size(k));
                %                 return;
                % 			else
                % 				rethrow(err);
                % 			end
                %         end
                if numel(varargin) >= 1 && isnumeric(varargin{1}) && ismatrix(varargin{1})
                    [ymu(RVmask),ys2(RVmask)] = predict(self, [xs, varargin{1}(RVmask,:)], varargin{2:end});
                else
                    [ymu(RVmask),ys2(RVmask)] = predict(self, xs, varargin{:});
                end
            end
        end
        
        function varargout = predict(self, xs, varargin)
            % Similar to GP.predict, but with optional normalization option
            % of the output.
            
            N = max(nargout, 1);
            varargout = cell(1, N);
            
            [varargout{:}] = predict@nextgp.GP(self, xs);
            
            if nargin > 2 && ~isempty(varargin{1})
                if strcmpi(varargin{1},'normalized') || strcmpi(varargin{1},'n')
                    return;
                else
                    error(['unknown option: ' varargin{1}]);
                end
            end
            
            if self.signals.normalize
                varargout{1} = varargout{1}*self.signals.m_std.(self.signals.m_O)+self.signals.m_mean.(self.signals.m_O);
                if N > 1
                    varargout{2} = varargout{2}*(self.signals.m_std.(self.signals.m_O)^2);
                end
            end
        end
        
        function optimizePrior(self)
            if isempty(self.gpdata.training_data.x)
                return;
            end
            if self.hypOptim.enable
                [newhyps, ~] = feval(self.hypOptim.optimizer,...
                    self.gpml_hyp, @gp, self.hypOptim.iter,...
                    self.gpml_inf,self.gpml_mean, self.gpml_cov, self.gpml_lik,...
                    self.gpdata.training_data.x, self.gpdata.training_data.y);
                self.resetPrior(newhyps);
            else
                warning('Hyperparameter optimization is configured to be disabled! Nothing to be done. Leaving...');
            end
        end
        
        function reduce(self, informationGain)
            % Remove points with low information gain
            
            if isempty(self.gpdata.training_data.x), return; end
            
            exceededSize = self.m_size - self.reducing.maxSize;
            if exceededSize > 0
                % get sorted information gains
                if nargin == 1
                    informationGain = self.getInformationGain();
                end
                id = informationGain(end-exceededSize+1:end,2);    % the indices of the points in training dataset to be removed
                self.gpdata.training_data.x(id, :) = [];
                self.gpdata.training_data.y(id, :) = [];
                self.BVtst(id,:) = [];
                self.m_size = size(self.BVtst, 1);
                
                self.gpdata.updatePosterior();
                %~ fprintf('reduced data timestamps: %s\n',num2str(timestamps(id)));
                
                %{
                for i = 1:length(id)
                    self.reducing.indexLifes(id(i),2) = self.signals.time(id(i));
                end
                %}
            end
            self.m_size = size(self.gpdata.training_data.x, 1);
        end
        
        function informationGain = getInformationGain(self)
            % Compute the information gain for the current training data
            % points.
            % Returns the informationGain matrix of size (Nx2) where the
            % first column contains the information gain values, and the
            % second column contains the indices of the training points.
            % The matrix is sorted in descending order of the first column.
            
            tst = self.BVtst;
            informationGain = NaN(self.m_size,2);
            informationGain(:,2) = 1:self.m_size;   % the indices of points in the training set, from 1
            
            switch self.reducing.type(1:3)
                case 'max'
                    sig = 1;
                    reducingmethod = self.reducing.type(4:end);
                case 'min'
                    sig = -1;
                    reducingmethod = self.reducing.type(4:end);
                otherwise
                    sig = 1; % the default prefix is "max", the preposition for maximum value
                    reducingmethod = self.reducing.type;
            end
            
            switch reducingmethod
                case 'likelihood'                    
                    for i = 1:self.m_size
                        [~,nlZ] = feval(self.gpml_inf{:}, self.gpml_hyp, self.gpml_mean, self.gpml_cov, self.gpml_lik, self.gpdata.training_data.x([1:i-1,i+1:end],:), self.gpdata.training_data.y([1:i-1,i+1:end]));
                        informationGain(i,1) = -nlZ*sig;
                    end
                case 'optimizedlikelihood'
                    for i = 1:self.m_size
                        [~, nlZ] = feval(self.hypOptim.optimizer,...
                            self.gpml_hyp, @gp, self.hypOptim.iter,...
                            self.gpml_inf, self.gpml_mean, self.gpml_cov, self.gpml_lik,...
                            self.gpdata.training_data.x([1:i-1,i+1:end],:), self.gpdata.training_data.y([1:i-1,i+1:end]));
                        informationGain(i,1) = -nlZ(end)*sig;
                    end
                case 'euclid'
                    % sorts the elements of active set by ascending euclid distance between them
                    D = sq_dist([self.gpdata.training_data.x, self.gpdata.training_data.y]')*sig;
                    
                    % set the diagonal to NaN so they won't affect the
                    % result
                    D(1:(self.m_size+1):end) = NaN;
                    
                    informationGain(:,1) = min(D,[],2) * sig;
                    
                    %{
                    D1 = zeros(self.m_size*(self.m_size-1)/2, 2);  % first column = distance, second column = index
                    
                    D_id2 = repmat(tst(1:self.m_size),1,self.m_size);
                    D_id1 = D_id2';
                    diagsizes = (self.m_size-1:-1:1);
                    diagids = [0 cumsum(diagsizes)];
                    D12 = NaN(2*diagids(end),3);
                    k = 1;
                    for i = 1:length(diagsizes)
                        D12((diagids(i)+1):diagids(i+1),:) = [diag(D,i),diag(D_id1,i),diag(D_id2,i)];
                    end
                    D1 = [D12(:,[1,3]);D12(:,[1,2])];
                    [~,D1i] = sort(D1(:,1),'descend');
                    D1 = D1(D1i,:);
                    %~ if sig>0 occurrence='last'; else  occurrence='first'; end;
                    occurrence='last';
                    [informationGain(:,2),D1indices]=unique(D1(:,2),occurrence);
                    informationGain(:,1)=D1(D1indices,1);
                    %~ [min(D1(:,1)),min(informationGain(:,1))]
                    %}
                case 'linearindependence'
                    % sorts the elements of active set by ascending linear
                    % independence in RKHS
                    [~, se2] = self.predict(self.gpdata.training_data.x);
                    informationGain(:,1) = sqrt(se2);
                case 'windowing'
                    % sorts the elements of active set by ascending euclid distance between them
                    informationGain(:,1) = tst(:);
                otherwise
                    error(['unknown reducing method: ' reducingmethod]);
            end
            
            informationGain = self.applyForgetting(informationGain);

            % Sort the matrix
            [~, sid] = sort(informationGain(:,1), 'descend');
            informationGain = informationGain(sid,:);

        end
        
        function informationGain = applyForgetting(self, informationGain)
            %
            %
            % InformationGain is a (Nx2) vector of information gains with
            % corresponding indices in the training dataset.
            %
            % apply a forgetting (by sample age) term to an already computed
            % information gain for each sample inside dataset. The forgetting
            % factor is linear with correpsonding sample timestamps.
            
            % shift information gain to be equal or greater than 0
            informationGain(:,1) = informationGain(:,1) - min(informationGain(:,1));
            
            % get the timestamps of the corresponding indices in the
            % informationGain
            tst = reshape(self.BVtst(informationGain(:,2)),[],1);
            timestampnow = max(tst);
            
            switch self.forgetting.type
                case 'linear'
                    informationGain(:,1) = informationGain(:,1) - self.forgetting.factor.*(timestampnow-tst);
                case 'exponential'
                    informationGain(:,1) = informationGain(:,1).*self.forgetting.factor.^(timestampnow-tst);
                case 'none'
                    % do nothing
                otherwise
                    error('Unknown forgetting type: %s', self.forgetting.type);
            end
        end
        
        function new = copy(self)
            % Copy super_prop
            new = feval(class(self),self.signals);
            
            % Copy all non-hidden properties.
            p = properties(self);
            for i = 1:length(p)
                new.(p{i}) = self.(p{i});
            end
            new.signals = self.signals.copy();
        end
    end
    
end




%~ egp.Ts=Ts; %->to dyn
%~ egp.lambda=lambda; %-> MPC control
%~ egp.eta=eta; %-> MPC control
%~ egp.rho=rho; %-> MPC control
%~ egp.rhoT=rhoT; %-> MPC control
%~ egp.RegulatorEProfile=RegulatorEProfile; %-> MPC control
%~ egp.RegulatorVProfile=RegulatorVProfile; %-> MPC control
%~ egp.RegulatorUProfile=RegulatorUProfile; %-> MPC control
%~ egp.uexp=uexp; %-> MPC control

%~ egp.msteps=msteps;%% -> MPC control
%~ egp.upd_steps=upd_steps; %% -> to dyn, specific for update condition

%~ egp.udelta=process.udelta; %% lacks impl. for multi-input
%~ egp.umean=process.umean; %% lacks impl. for multi-input
%~ egp.ydelta=process.ydelta; %% lacks impl. for multi-output?
%~ egp.ymean=process.ymean; %% lacks impl. for multi-output?
%~ egp.BVt_id=[];%% -> change name to timestamps: egp.BVtst
