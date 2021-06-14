function main()
    close all
    simulation = targetSimulation();
    
    simulation.getData();
    simulation.exportData(1,'w','test_test1.csv');
    for k = 1:10
        simulation.getData();
        simulation.exportData(k+1,'a','test_test1.csv');
    end
    
    
end

