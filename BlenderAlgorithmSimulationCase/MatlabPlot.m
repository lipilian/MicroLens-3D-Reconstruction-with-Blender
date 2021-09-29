addpath('./npy-matlab');
data0 = readNPY('z0Points.npy'); data0 = data0(2:end,:);
data1 = readNPY('z-0.02Points.npy'); data1 = data1(2:end,:);
data2 = readNPY('z0.02Points.npy'); data2 = data2(2:end,:);
data = [data0;data1;data2];
data(:,1) = data(:,1) + 0.188760012388229;
data = data * 1000;
figure()
scatter3(data(:,2),data(:,3),data(:,1),'filled','k');
axis equal;
zlim([-20.001,20.001]);
zticks([-20,0,20]);
set(gcf,'color','w');
set(gca,'FontSize',18);

%% 
clear all;
load('./temp/matlabData.mat')
ref = [ref1; ref2; ref3];
refo = [refo1; refo2; refo3];
figure()
scatter3(ref(:,1) - 17.95,ref(:,2) - 11.95,ref(:,3),'filled','k');
hold on;
scatter3(refo(:,1) - 17.95,refo(:,2)-11.95,refo(:,3),'filled','r');
axis equal;
zlim([-26,26]);
zticks([-26,0,26]);
xticks([-5,0,5]);
yticks([-5,0,5]);
set(gcf,'color','w');
set(gca,'FontSize',18);