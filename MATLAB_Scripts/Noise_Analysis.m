%% Load data

clear all

filepath = '/Users/lijiaming/Library/CloudStorage/OneDrive-UW/UW/DNA_Tag/Results/Error_Count_20230330.xlsx';

[num,txt,raw] = xlsread(filepath);

dates = num(:,1);
bits = num(:,2);
errors = num(:,3);
txt = txt(2:end,4:end);

analogpath = '/Users/lijiaming/Library/CloudStorage/OneDrive-UW/UW/DNA_Tag/Results/Alex/DNATag_Raw_Data/';

folders = dir(analogpath);
analog_raw = {};        %Column 1 = encoded bits, columne 2 = normalized signal decays

j = 1;
for i = 1:length(folders)
    if folders(i).name(1) == '2'
        
        analog_raw{j,1} = readmatrix([analogpath folders(i).name '/' folders(i).name '.csv']);

        analog_raw{j,2} = csvread([analogpath folders(i).name '/' ...
            folders(i).name '_Decay.csv']);
        j = j+1;
    end
end


fprintf('Data loaded!\n');

colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F',...
    '#FF0000','#00FF00','#0000FF','#00FFFF','#FF00FF','#FFFF00','#000000'};


%% 20231211: Errors of oligo concentration normalized VS not

b = unique(bits);
n = [];

errs = cell(3,1);  % solution, dried w TreHL, dried wo TreHL
err_mean = [];
err_v = [];
err_p = [];
err_min = [];
err_max = [];

p = [];
h = [];

for i = [24, 25]
    n = [n;find(bits == i)];
end


for j = 1:length(n)
    if (txt{n(j),2} == 'N')
        errs{1} = [errs{1};errors(n(j))];
    elseif (txt{n(j),2} == 'Y')
            if (txt{n(j),3} == 'Y')
                errs{2} = [errs{2};errors(n(j))];
            else
                if (txt{n(j),4} == 'DS')
                    errs{3} = [errs{3};errors(n(j))];
                end
            end
    end
end







%% 20231203: error of 24/25-bits, solution VS dried w/ trehalose VS wo trehalose

% txt = {normalized, dried, w TreHL, ds/ss}

b = unique(bits);
n = [];

errs = cell(3,1);  % solution, dried w TreHL, dried wo TreHL
err_mean = [];
err_v = [];
err_p = [];
err_min = [];
err_max = [];

p = [];
h = [];

for i = [24, 25]
    n = [n;find(bits == i)];
end

for j = 1:length(n)
    if (txt{n(j),2} == 'N')
        errs{1} = [errs{1};errors(n(j))];
    elseif (txt{n(j),2} == 'Y')
            if (txt{n(j),3} == 'Y')
                errs{2} = [errs{2};errors(n(j))];
            else
                if (txt{n(j),4} == 'DS')
                    errs{3} = [errs{3};errors(n(j))];
                end
            end
    end
end

for i = 1:3
    err_mean(i) = mean(errs{i});
    err_v(i) = std(errs{i});
    err_min(i) = min(errs{i});
    err_max(i) = max(errs{i});
    
end

[h(1),p(1)] = ttest2(errs{1},errs{2});
[h(2),p(2)] = ttest2(errs{1},errs{3});

%% Plot
figure 
hold on
xdummy(1:2:35) = [0:0.05:0.85];
xdummy(2:2:34) = [-0.05:-0.05:-0.85];

x = cell(size(errs));
for i = 1:3
    x{i} = xdummy(1:length(errs{i}))+i;
    scatter(x{i},errs{i});
end
xx = [1:3];

errorbar(xx,err_mean,err_v,'LineStyle','none','Marker','d','MarkerSize',10);
xlim([0.5,3.5]);

text(1.5,7,['p1 = ', num2str(p(1)),', p2 = ',num2str(p(2))]);


hold off
% print('/Users/lijiaming/Library/CloudStorage/OneDrive-UW/UW/DNA_Tag/Manuscripts/NatComm_2023/Figures/Source/errbar_p.pdf','-dpdf');

             
                
            
        
        
        

%% Error VS Bit #, excluding dried wo/ TreHL

% txt = {normalized, dried, w TreHL, ds/ss}

b = unique(bits);

err_by_bit = {};
err_mean = [];
err_v = [];
err_min = [];
err_max = [];

for i = 1:length(b)
    tempn = find(bits==b(i));
    n = tempn;
    for j = 1:length(tempn)
        if (txt{tempn(j),2} == 'Y') && (txt{tempn(j),3} == 'N')
            n(find(n==tempn(j))) = [];
            
        else if txt{tempn(j),4} == 'SS'
            n(find(n==tempn(j))) = [];
            end
        end
    end
    
    temp = errors(n);
    length(temp)
    err_by_bit{i} = temp;
    err_mean(i) = mean(temp);
    err_v(i) = std(temp);
    err_min(i) = min(temp);
    err_max(i) = max(temp);
    
end
err_v = err_v';
err_mean = err_mean';
err_by_bit = err_by_bit';


savepath = '/Users/lijiaming/Library/CloudStorage/OneDrive-UW/UW/DNA_Tag/Manuscripts/NSDI_23/Figures/';

x = [err_by_bit{length(b)-1};err_by_bit{length(b)-2}];      % 24&25 bits
confidence = [mean(x) + 2*std(x),mean(x) - 2*std(x)];

figure
hold on
errorbar(b, err_mean,err_v,'-s','MarkerSize',8,'MarkerEdgeColor','red','MarkerFaceColor','red');
axis([0,48,0,4]);
box on;
text(0,3,['95% confidence interval for 24&25 bits: µ+2*std = ' ...
    num2str(confidence(1)), ', ',num2str(confidence(2))]);
xlabel('Bit #');
ylabel('Errors');
set(gcf,'units','normalized','position',[0.4,0.1,0.3,0.4]);
hold off
print([savepath 'ErrBar_' datestr(datetime)],'-dpdf');


%% Noised dependent on denaturation during dehydration

sample = [find(bits == 24); find(bits==25)];

for i = 1:size(txt,1)
    if txt{i,4} == 'SS'
        sample = sample(find(sample~=i));
    end
end


sample = sort(sample);

sample_err = cell(2,1);        % {Dried wo TreHL; Dried w TreHL}


analog1 = cell(2,1);
analog0 = cell(2,1);

for i = 1:length(sample)
    n = sample(i);
    
    if txt{n,2}=='Y' % && txt{n,1} == 'Y'
        if txt{n,3} == 'N'
            sample_err{1} = [sample_err{1};errors(n)];
            a1 = analog_raw{n,2}(analog_raw{n,1}==1);
            analog1{1} = [analog1{1}; a1(:)];
            a0 = analog_raw{n,2}(analog_raw{n,1}==0);
            analog0{1} = [analog0{1}; a0(:)];
            
        else
            sample_err{2} = [sample_err{2};errors(n)];
            
            a1 = analog_raw{n,2}(analog_raw{n,1}==1);
            analog1{2} = [analog1{2}; a1(:)];
            a0 = analog_raw{n,2}(analog_raw{n,1}==0);
            analog0{2} = [analog0{2}; a0(:)];
            
            
        end
    end

end

savepath = '/Users/lijiaming/Library/CloudStorage/OneDrive-UW/UW/DNA_Tag/Manuscripts/NSDI_23/Figures/';


% figure
% hold on
% title('Not normalized');
% yyaxis left
% histogram(sample_err{1,1});
% yyaxis right
% histogram(sample_err{1,2});
% legend('Dried wo TreHL','Dried w TreHL');
% t = ['Mean 1 = ' num2str(mean(sample_err{1})) ', Mean 2 = ' ...
%     num2str(mean(sample_err{2}))];
% text(1,1,t);
% xlabel('Errors');
% ylabel('Readout Counts');
% box on
% hold off
% print([savepath 'NoiseTreHL_' datestr(datetime)],'-dpdf');

figure

subplot(1,2,1)

hold on
title('Analog, Dried wo TreHL');
yyaxis right
% ylim([0,floor(length(analog1{1})/4)]);
histogram(analog1{1},[0:0.1:1]);
yyaxis left
% ylim([0,floor(length(analog0{1})/4)]);
histogram(analog0{1},[0:0.1:1]);
legend('1','0');
box on
hold off

subplot(1,2,2)
hold on
title('Analog, Dried w TreHL');
yyaxis right
% ylim([0,floor(length(analog1{2})/2)]);
histogram(analog1{2},[0:0.1:1]);
yyaxis left
% ylim([0,floor(length(analog0{2})/2)]);
histogram(analog0{2},[0:0.1:1]);
legend('1','0');
xlabel('Normalized Signal Decay');
ylabel('Spot Counts');
box on
hold off


%% Noised dependent on concentration normalization during dehydration
% txt = {normalized, dried, w TreHL, ds/ss}

sample = [find(bits == 24); find(bits==25)];

for i = 1:size(txt,1)
    if txt{i,4} == 'SS'
        sample = sample(find(sample~=i));
    end
end

sample = sort(sample);

sample_err = cell(2,1);        % {Dried wo TreHL; Dried w TreHL}
analog1 = cell(2,1);
analog0 = cell(2,1);


for i = 1:length(sample)
    n = sample(i);
    
    if txt{n,2}=='N' % && txt{n,1} == 'Y'
        if txt{n,1} == 'N'
            sample_err{1} = [sample_err{1};errors(n)];
            a1 = analog_raw{n,2}(analog_raw{n,1}==1);
            analog1{1} = [analog1{1}; a1(:)];
            a0 = analog_raw{n,2}(analog_raw{n,1}==0);
            analog0{1} = [analog0{1}; a0(:)];
        else
            sample_err{2} = [sample_err{2};errors(n)];
            a1 = analog_raw{n,2}(analog_raw{n,1}==1);
            analog1{2} = [analog1{2}; a1(:)];
            a0 = analog_raw{n,2}(analog_raw{n,1}==0);
            analog0{2} = [analog0{2}; a0(:)];
            
        end
    end

end


savepath = '/Users/lijiaming/Library/CloudStorage/OneDrive-UW/UW/DNA_Tag/Manuscripts/NSDI_23/Figures/';


% figure
% hold on
% title('Not normalized');
% histogram(sample_err{1,1});
% histogram(sample_err{1,2});
% legend('Not normalized','Normalized');
% t = ['Mean 1 = ' num2str(mean(sample_err{1})) ', Mean 2 = ' ...
%     num2str(mean(sample_err{2}))];
% text(1,1,t);
% xlabel('Errors');
% ylabel('Readout Counts');
% box on
% hold off
% print([savepath 'Normalized_' datestr(datetime)],'-dpdf');

figure

subplot(1,2,1)

hold on
title('Analog, not normalized');
yyaxis right
% ylim([0,floor(length(analog1{1})/3)]);
histogram(analog1{1},[0:0.1:1]);
yyaxis left
% ylim([0,floor(length(analog0{1})/3)]);
histogram(analog0{1},[0:0.1:1]);
text()
legend('1','0');
box on
hold off

subplot(1,2,2)
hold on
title('Analog, Normalized');
yyaxis right
% ylim([0,floor(length(analog1{2})/3)]);
histogram(analog1{2},[0:0.1:1]);
yyaxis left
% ylim([0,floor(length(analog0{2})/3)]);
histogram(analog0{2},[0:0.1:1]);
legend('1','0');
xlabel('Normalized Signal Decay');
ylabel('Spot Counts');
box on
hold off

%% Analysis on analog noise, gaussian fitting

analog1 = [];
analog0 = [];

for i = 1:length(analog_raw)
    if txt{i,2}=='Y'  && txt{i,3} == 'N' && prod(txt{i,4} == 'DS')
        continue
    end
    
    a1 = analog_raw{i,2}(analog_raw{i,1}==1);
    analog1 = [analog1;a1(:)];
    a0 = analog_raw{i,2}(analog_raw{i,1}==0);
    analog0 = [analog0;a0(:)];
end

figure
title('Analog of All');
subplot(1,3,1)
hold on
title('Analog of All');
histogram(analog1,[0:0.1:1],'FaceColor',colors{1});

x = analog1;
m = mean(x,'all');
d = std(x,'omitnan');
xx = [0:0.01:1];
ylimit = get(gca,'ylim');
y = 1/(d*sqrt(2*pi))*exp( -1/2*((xx-m)/d).^2 )*length(x);
y = y * ylimit(2)/max(y);
plot(xx,y);
text(0.2,0.2*ylimit(2),['Mean = ' num2str(m) ', Std = ' num2str(d)]);

box on
hold off
subplot(1,3,2)
hold on
histogram(analog0,[0:0.1:1],'FaceColor',colors{2});

x = analog0;
m = mean(x,'all');
d = std(x,'omitnan');
xx = [0:0.01:1];
ylimit = get(gca,'ylim');
y = 1/(d*sqrt(2*pi))*exp( -1/2*((xx-m)/d).^2 )*length(x);
y = y * ylimit(2)/max(y);
plot(xx,y);
text(0.2,0.2*ylimit(2),['Mean = ' num2str(m) ', Std = ' num2str(d)]);

box on
hold off
subplot(1,3,3)
hold on
yyaxis right
histogram(analog1,[0:0.1:1]);
yyaxis left
histogram(analog0,[0:0.1:1]);
legend('1', '0');
box on
hold off

%% Analysis on analog noise, symmetric Gaussian fitting for analog 0

analog1 = [];
analog0 = [];

for i = 1:length(analog_raw)
    if txt{i,2}=='Y'  && txt{i,3} == 'N' && prod(txt{i,4} == 'DS')
        continue
    end
    
    a1 = analog_raw{i,2}(analog_raw{i,1}==1);
    analog1 = [analog1;a1(:)];
    a0 = analog_raw{i,2}(analog_raw{i,1}==0);
    analog0 = [analog0;a0(:)];
end

figure
title('Analog of All');
subplot(1,3,1)
hold on
title('Analog of All');
histogram(analog1,[0:0.1:1],'FaceColor',colors{1});

x = analog1;
m = mean(x,'all');
d = std(x,'omitnan');
xx = [0:0.01:1];
ylimit = get(gca,'ylim');
y = 1/(d*sqrt(2*pi))*exp( -1/2*((xx-m)/d).^2 )*length(x) / (d * (normcdf((1-m)/d) - normcdf(-m/d)));
y = y * ylimit(2)/max(y);
plot(xx,y);
text(0.2,0.2*ylimit(2),['Mean = ' num2str(m) ', Std = ' num2str(d)]);

box on
hold off
subplot(1,3,2)
hold on
histogram(analog0,[0:0.1:1],'FaceColor',colors{2});

x = [analog0;-analog0];
m = mean(x,'all');
% m = 0.07;
d = std(x,'omitnan');
xx = [0:0.01:1];
ylimit = get(gca,'ylim');
y = 1/(d*sqrt(2*pi))*exp( -1/2*((xx-m)/d).^2 )*length(x) ;
y = y * ylimit(2)/max(y);
y = y * 200 /max(y);
plot(xx,y);
text(0.2,0.2*ylimit(2),['Mean = ' num2str(m) ', Std = ' num2str(d)]);

box on
hold off
subplot(1,3,3)
hold on
yyaxis right
histogram(analog1,[0:0.1:1]);
yyaxis left
histogram(analog0,[0:0.1:1]);
legend('1', '0');
box on
hold off


%% Analysis on analog noise, Poisson fitting

analog1 = [];
analog0 = [];

for i = 1:length(analog_raw)
    if txt{i,2}=='Y'  && txt{i,3} == 'N' && prod(txt{i,4} == 'DS')
        continue
    end
    
    a1 = analog_raw{i,2}(analog_raw{i,1}==1);
    analog1 = [analog1;a1(:)];
    a0 = analog_raw{i,2}(analog_raw{i,1}==0);
    analog0 = [analog0;a0(:)];
end

figure
title('Analog of All');
subplot(1,3,1)
hold on
title('Analog of All');
histogram(analog1,[0:0.1:1],'FaceColor',colors{1});

x = analog1;
m = mean(x,'all');
d = std(x,'omitnan');
xx = [0:0.01:1];
ylimit = get(gca,'ylim');
y = 1/(d*sqrt(2*pi))*exp( -1/2*((xx-m)/d).^2 )*length(x) / (m * (normcdf((1-m)/d) - normcdf(-m/d)));
y = y * ylimit(2)/max(y);
plot(xx,y);
text(0.2,0.2*ylimit(2),['Mean = ' num2str(m) ', Std = ' num2str(d)]);

box on
hold off
subplot(1,3,2)
hold on
histogram(analog0,[0:0.1:1],'FaceColor',colors{2});

x = analog0;
m = mean(x,'all');
d = std(x,'omitnan');
xx = [0:0.01:1];
ylimit = get(gca,'ylim');
y = 1/(d*sqrt(2*pi))*exp( -1/2*((xx-m)/d).^2 )*length(x) / (m * (normcdf((1-m)/d) - normcdf(-m/d)));
y = y * ylimit(2)/max(y);
plot(xx,y);
text(0.2,0.2*ylimit(2),['Mean = ' num2str(m) ', Std = ' num2str(d)]);

box on
hold off
subplot(1,3,3)
hold on
yyaxis right
histogram(analog1,[0:0.1:1]);
yyaxis left
histogram(analog0,[0:0.1:1]);
legend('1', '0');
box on
hold off



%% Analysis on the 20230403 Merging experiment

a1 = cell(6,1);        % 1, 0+1, 1+1, 1+0+0, 1+1+0, 1+1+1
a0 = cell(3,1);        % 0, 0+0, 0+0+0

tempraw = analog_raw(69:74,:);

for i = 1:3     % 1 and 0
    
    a = tempraw{i,2}(tempraw{i,1}==1);
    a1{1} = [a1{1};a(:)];
    
    a = tempraw{i,2}(tempraw{i,1}==0);
    a0{1} = [a0{1};a(:)];
    
end

for i = 4:5
    
    if i == 4
        mergedbits = tempraw{1,1} + tempraw{2,1};
    else
        mergedbits = tempraw{1,1} + tempraw{3,1};
    end
    a = tempraw{i,2}(mergedbits == 1);      % 0+1
    a1{2} = [a1{2};a(:)];
    
    a = tempraw{i,2}(mergedbits == 2);      % 1+1
    a1{3} = [a1{3};a(:)];
    
    a = tempraw{i,2}(mergedbits == 0);      % 0+0
    a0{2} = [a0{2};a(:)];
end

mergedbits = tempraw{1,1}+tempraw{2,1}+tempraw{3,1};

a = tempraw{6,2}(mergedbits == 1);      % 0+0+1
a1{4} = [a1{4};a(:)];

a = tempraw{6,2}(mergedbits == 2);      % 1+1+0
a1{5} = [a1{5};a(:)];

a = tempraw{6,2}(mergedbits == 3);      % 1+1+1
a1{6} = [a1{6};a(:)];

a = tempraw{6,2}(mergedbits == 0);      % 0+0+0
a0{3} = [a0{3};a(:)];



%% Analysis on the 20230406 Merging experiment

a1 = cell(6,1);        % 1, 0+1, 1+1, 1+0+0, 1+1+0, 1+1+1
a0 = cell(3,1);        % 0, 0+0, 0+0+0

tempraw = analog_raw(75:80,:);

for i = 1:3     % 1 and 0
    
    a = tempraw{i,2}(tempraw{i,1}==1);
    a1{1} = [a1{1};a(:)];
    
    a = tempraw{i,2}(tempraw{i,1}==0);
    a0{1} = [a0{1};a(:)];
    
end

for i = 4:5
    
    if i == 4
        mergedbits = tempraw{1,1} + tempraw{2,1};
    else
        mergedbits = tempraw{1,1} + tempraw{3,1};
    end
    a = tempraw{i,2}(mergedbits == 1);      % 0+1
    a1{2} = [a1{2};a(:)];
    
    a = tempraw{i,2}(mergedbits == 2);      % 1+1
    a1{3} = [a1{3};a(:)];
    
    a = tempraw{i,2}(mergedbits == 0);      % 0+0
    a0{2} = [a0{2};a(:)];
end

mergedbits = tempraw{1,1}+tempraw{2,1}+tempraw{3,1};

a = tempraw{6,2}(mergedbits == 1);      % 0+0+1
a1{4} = [a1{4};a(:)];

a = tempraw{6,2}(mergedbits == 2);      % 1+1+0
a1{5} = [a1{5};a(:)];

a = tempraw{6,2}(mergedbits == 3);      % 1+1+1
a1{6} = [a1{6};a(:)];

a = tempraw{6,2}(mergedbits == 0);      % 0+0+0
a0{3} = [a0{3};a(:)];

%% Analysis 20230403 + 20230406 Merging experiments

a1 = cell(6,1);        % 1, 0+1, 1+1, 1+0+0, 1+1+0, 1+1+1
a0 = cell(3,1);        % 0, 0+0, 0+0+0

tempraw = cell(6,2);

for i = 1:6
    tempraw{i,1} = [analog_raw{i+68,1};analog_raw{i+74,1}];
    tempraw{i,2} = [analog_raw{i+68,2};analog_raw{i+74,2}];
end

for i = 1:3     % 1 and 0
    
    a = tempraw{i,2}(tempraw{i,1}==1);
    a1{1} = [a1{1};a(:)];
    
    a = tempraw{i,2}(tempraw{i,1}==0);
    a0{1} = [a0{1};a(:)];
    
end

for i = 4:5
    
    if i == 4
        mergedbits = tempraw{1,1} + tempraw{2,1};
    else
        mergedbits = tempraw{1,1} + tempraw{3,1};
    end
    a = tempraw{i,2}(mergedbits == 1);      % 0+1
    a1{2} = [a1{2};a(:)];
    
    a = tempraw{i,2}(mergedbits == 2);      % 1+1
    a1{3} = [a1{3};a(:)];
    
    a = tempraw{i,2}(mergedbits == 0);      % 0+0
    a0{2} = [a0{2};a(:)];
end

mergedbits = tempraw{1,1}+tempraw{2,1}+tempraw{3,1};

a = tempraw{6,2}(mergedbits == 1);      % 0+0+1
a1{4} = [a1{4};a(:)];

a = tempraw{6,2}(mergedbits == 2);      % 1+1+0
a1{5} = [a1{5};a(:)];

a = tempraw{6,2}(mergedbits == 3);      % 1+1+1
a1{6} = [a1{6};a(:)];

a = tempraw{6,2}(mergedbits == 0);      % 0+0+0
a0{3} = [a0{3};a(:)];



%% Plot

combinations = {'1','1+0','1+1','1+0+0','1+1+0','1+1+1'};

colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F',...
    '#FF0000','#00FF00','#0000FF','#00FFFF','#FF00FF','#FFFF00','#000000'};


figure
for i = 1:length(a1)
    subplot(7,1,i)
    title(combinations{i});
    histogram(a1{i},[0:0.1:1]);
    x = mean(a1{i});
    xline(x,'--r','LineWidth',2);
    text(0.2,1,['Mean = ' num2str(x)]);
    legend(combinations{i});
end

subplot(7,1,7)
hold on
for i = 1:length(a0)
    histogram(a0{i},[0:0.1:1])
end
box on
legend('0','0+0','0+0+0');
hold off

% 
% figure
% subplot(3,1,1)
% hold on
% for i = 1:3
%     histogram(a1{i},[0:0.1:1],'FaceColor',colors{i});
% end
% legend('1','0+1','1+1');
% subplot(3,1,2)
% hold on
% for i = 4:length(a1)
%     histogram(a1{i},[0:0.1:1],'FaceColor',colors{i});
% end
% legend('0+0+1','1+1+0','1+1+1');
% 
% hold off
% 
% subplot(3,1,3)
% hold on
% for i = 1:length(a0)
%     histogram(a0{i},[0:0.1:1])
% end
% box on
% legend('0','0+0','0+0+0');
% hold off




    





