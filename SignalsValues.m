classdef SignalsValues < handle
    %SIGNALSVALUES Class to store signal values for SignalsModel.
    %   Each signal has a unique name, which is a field / property of an
    %   instance of this class. Multiple SignalsModel objects can share the
    %   same SignalsValues storage.
    
    properties (Access=private)
        m_curSize = 0; % current size of signal vectors
        m_growstep = 1024; % how to grow vectors
        m_signals = struct;
    end
    
    methods
        function self = SignalsValue(growstep)
            if nargin > 0
                assert(growstep > 0);
                self.m_growstep = growstep;
            end
        end
        
        function b = isfield(self, fldnames)
            b = isfield(self.m_signals, fldnames);
        end
        
        function grow(self, newsize)
            % Grow all signal vectors to at least the new size.
            if self.m_curSize >= newsize, return; end
            newsize = ceil(newsize/self.m_growstep)*self.m_growstep;
            self.m_curSize = newsize;
            allprops = fieldnames(self.m_signals);
            for k = 1:numel(allprops)
                self.m_signals.(allprops{k})(end+1:newsize) = NaN;
            end
        end
        
        function addSignal(self, name, values)
            % Add a new signal with given name and optionally initial
            % values
            assert(~isfield(self.m_signals, name), 'Signal %s already exists.', name);
            
            if nargin > 2
                values = values(:);
                newsize = length(values);
                self.grow(newsize);
                self.m_signals.(name) = NaN(self.m_curSize, 1);
                self.m_signals.(name)(1:newsize) = values;
            else
                self.m_signals.(name) = NaN(self.m_curSize, 1);
            end
        end
        
        function v = getSignal(self, name, idx)
            % Get values of given signal with given index idx (if omitted
            % then the entire signal vector).
            if nargin < 3
                v = self.m_signals.(name);
            else
                v = self.m_signals.(name)(idx);
            end
        end
        
        function setSignal(self, name, idx, v)
            % Set values of given signal with given index idx (if empty
            % then the entire signal vector).
            if isempty(idx)
                self.m_signals.(name) = v;
            else
                % grow if necessary
                if islogical(idx)
                    self.grow(numel(idx));
                else
                    self.grow(max(idx(:)));
                end
                self.m_signals.(name)(idx) = v;
            end
        end
    end
    
end

