
thres = 0.38;

b = zeros(size(a));
b(a>thres)=1;

figure
hold on
imagesc(a);
colorbar;
axis([0.5,5.5,0.5,5.5]);
box on;
colormap(cool);
set(gcf,'units','normalized','position',[0.1,0.1,0.17,0.2]);
set(gca,'YDir','reverse');
hold off

figure
hold on
imagesc(b)
colormap(gray);
colorbar;
axis([0.5,5.5,0.5,5.5]);
box on;
set(gcf,'units','normalized','position',[0.4,0.1,0.17,0.2]);
set(gca,'YDir','reverse');
text(4,4,['Thres = ' num2str(thres)],'Color','r','FontSize',14);
hold off



