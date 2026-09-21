FFT_h1 = [0.006275 0.004603 0.009392 0.000113 0.012561];
FFT_h4 = [0.006771 0.001582 0.008617 0.000083 0.014352];

FFT = [FFT_h1; FFT_h4]';

%% =====================================================
% 95% Bootstrap CI
%% =====================================================

CI_h1 = [...
    0.005863 0.006650;
    0.001153 0.014886;
    0.007970 0.010649;
    0.000101 0.000145;
    0.010968 0.014500];

CI_h4 = [...
    0.006318 0.007109;
    0.001068 0.002436;
    0.007748 0.009767;
    0.000072 0.000101;
    0.013058 0.015856];

errLow_h1 = FFT_h1 - CI_h1(:,1)';
errUp_h1  = CI_h1(:,2)' - FFT_h1;

errLow_h4 = FFT_h4 - CI_h4(:,1)';
errUp_h4  = CI_h4(:,2)' - FFT_h4;

%% =====================================================
% FIGURE
%% =====================================================

figure('Color','w')

b = barh(FFT,'grouped');

% Colores Q1
b(1).FaceColor = [0.12 0.28 0.55];
b(2).FaceColor = [0.10 0.45 0.40];

b(1).EdgeColor = 'none';
b(2).EdgeColor = 'none';

set(gca,'FontSize',18,...
        'LineWidth',2,...
        'TickLabelInterpreter','latex')

set(gca,'YTickLabel',{'LSTM','Elman','GRU','NARX','ESN'})
set(gca,'YDir','reverse')

xlabel('\fontsize{12}{0}\selectfont FFT Error',...
       'Interpreter','latex',...
       'FontWeight','bold')

title('\fontsize{14}{0}\selectfont FFT Error Comparison Across RNN Architectures',...
      'Interpreter','latex',...
      'FontWeight','bold')

legend('h$_1$','h$_4$',...
       'Interpreter','latex',...
       'Location','best')

grid on
box on
hold on

xmax = max(FFT(:));

xlim([0 xmax*1.4])

set(gca,'FontSize',18,...
        'LineWidth',2,...
        'TickLabelInterpreter','latex')

%% =====================================================
% POSICIÓN DE LAS BARRAS
%% =====================================================

nModels = size(FFT,1);
nSeries = size(FFT,2);

groupWidth = 0.6;

for i = 1:nSeries

    y = (1:nModels) - groupWidth/2 + (2*i-1)*groupWidth/(2*nSeries);

    %% ============================================
    % Barras de error (95% BootCI)
    %% ============================================

    if i==1

        e1 = errorbar(FFT_h1,y,...
            errLow_h1,errUp_h1,...
            'horizontal','k',...
            'Color',[0.25 0.25 0.25],...
            'LineStyle','none',...
            'LineWidth',1.3,...
            'CapSize',8);

        e1.Annotation.LegendInformation.IconDisplayStyle='off';

    else

        e2 = errorbar(FFT_h4,y,...
            errLow_h4,errUp_h4,...
            'horizontal','k',...
            'Color',[0.25 0.25 0.25],...
            'LineStyle','none',...
            'LineWidth',1.3,...
            'CapSize',8);

        e2.Annotation.LegendInformation.IconDisplayStyle='off';

    end

    %% ============================================
    % Valores numéricos
    %% ============================================

    for j = 1:nModels

        val = FFT(j,i);

        if i==1
            offset = errUp_h1(j) + xmax*0.02;
        else
            offset = errUp_h4(j) + xmax*0.02;
        end

        text(val + offset,...
             y(j),...
             num2str(val,'%.6f'),...
             'FontSize',8,...
             'Color',[0.1 0.1 0.1],...
             'HorizontalAlignment','left',...
             'VerticalAlignment','middle');

    end

end

set(gca,'FontSize',10,'LineWidth',1)

hold off

%% =====================================================
% EXPORTAR PDF VECTORIAL
%% =====================================================

set(gcf,'Renderer','painters')

exportgraphics(gcf,...
               'FFT_plot.pdf',...
               'ContentType','vector')
