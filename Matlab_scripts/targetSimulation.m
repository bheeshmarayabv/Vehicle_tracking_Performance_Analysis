classdef targetSimulation < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        settings;
        groundTruth;
        measurement;
        dt;
        Tmax;
        t;
    end
    
    methods
        function obj = targetSimulation()
            obj.getSimulationSettings();
        end
        
        function obj = exportData(obj,run,mode,name)
            
            if mode == 'a'
                f = fopen(name,'a');
                
            else
                f = fopen(name,'w');
                
            end
            for k = 1:length(obj.t)

                fprintf(f,'%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n',run,obj.t(k),...
                        obj.measurement.r(k),obj.measurement.phi(k),obj.measurement.rr(k),obj.measurement.x(k),obj.measurement.y(k),obj.measurement.dx(k),obj.measurement.dy(k),...
                        obj.groundTruth.r(k),obj.groundTruth.phi(k),obj.groundTruth.rr(k),obj.groundTruth.x(k),obj.groundTruth.y(k),obj.groundTruth.dx(k),obj.groundTruth.dy(k));

            end
            fclose(f);
            
        end
        
        
        function obj = getMeasurement(obj)
            obj.measurement.r  = obj.groundTruth.r  + randn(size(obj.groundTruth.r))*obj.settings.sigma_m_r;
            obj.measurement.rr = obj.groundTruth.rr + randn(size(obj.groundTruth.rr))*obj.settings.sigma_m_rr;
            obj.measurement.a  = obj.groundTruth.a  + randn(size(obj.groundTruth.a))*obj.settings.sigma_m_a;
            obj.measurement.phi = obj.groundTruth.phi  + randn(size(obj.groundTruth.phi))*obj.settings.sigma_m_phi;
            obj.measurement.x  = (obj.measurement.r.*cos(obj.measurement.phi));
            obj.measurement.y  = (obj.measurement.r.*sin(obj.measurement.phi));
            obj.measurement.dx = (obj.measurement.rr.*cos(obj.measurement.phi));
            obj.measurement.dy = (obj.measurement.rr.*sin(obj.measurement.phi));
            
            
            
        end
        
       
        function obj = getSimulationSettings(obj)
            
            obj.dt = 1;
            obj.Tmax = 100;
            obj.settings.sigma_m_r = 5;
            obj.settings.sigma_m_rr = 0.1;
            obj.settings.sigma_m_a = 1;
            obj.settings.sigma_m_phi = deg2rad(1);
            obj.settings.sigma_gt_a = 0.2;
            obj.settings.sigma_gt_phi_dot_dot = 0.0001;
            
            
        end
        
        function obj = getGroundTruth(obj)
            
            p = rand(3,1);
            P = sort(p);
            
            obj.groundTruth.a = zeros(size(obj.t));
            
            obj.groundTruth.a(obj.t < P(1)*obj.Tmax) = 0.2;
            obj.groundTruth.a(obj.t > P(2)*obj.Tmax) = -0.2;
            obj.groundTruth.a(obj.t > P(3)*obj.Tmax) = 0;
            
            obj.groundTruth.a = obj.groundTruth.a + randn(size(obj.t))*obj.settings.sigma_gt_a;
            obj.groundTruth.rr = cumsum(obj.groundTruth.a)*obj.dt;
            obj.groundTruth.r = cumsum(obj.groundTruth.rr)*obj.dt;
            
            obj.groundTruth.r = obj.groundTruth.r;
            p = max(abs(obj.groundTruth.r(obj.groundTruth.r<0)));
            if ~isempty(p)
                obj.groundTruth.r = obj.groundTruth.r + p;
            end
            
            p = rand(3,1);
            P = sort(p);
            obj.groundTruth.phidotdot = zeros(size(obj.t));
            obj.groundTruth.phidotdot(obj.t < P(1)*obj.Tmax) = 0.0001;
            obj.groundTruth.phidotdot(obj.t > P(2)*obj.Tmax) = -0.0001;
            obj.groundTruth.phidotdot(obj.t > P(3)*obj.Tmax) = 0;
            obj.groundTruth.phidotdot = obj.groundTruth.phidotdot ...
                + randn(size(obj.t))*obj.settings.sigma_gt_phi_dot_dot;
            obj.groundTruth.phidot = cumsum(obj.groundTruth.phidotdot)*obj.dt;
            
            obj.groundTruth.phi = cumsum(obj.groundTruth.phidot)*obj.dt;
            obj.groundTruth.x = obj.groundTruth.r .* cos(obj.groundTruth.phi);
            obj.groundTruth.y = obj.groundTruth.r .* sin(obj.groundTruth.phi);
            obj.groundTruth.dx = obj.groundTruth.rr .* cos(obj.groundTruth.phi);
            obj.groundTruth.dy = obj.groundTruth.rr .* sin(obj.groundTruth.phi);
        end
        
        function obj = getData(obj)
            obj.t = 0:obj.dt:obj.Tmax;
            obj.getGroundTruth();
            obj.getMeasurement();
        end
    end
end



