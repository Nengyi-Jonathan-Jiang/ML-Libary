package chart;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.CategoryTableXYDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class LineChart extends ApplicationFrame {

    private final CategoryTableXYDataset dataset;

    public LineChart() {
        super("Chart");

        dataset = new CategoryTableXYDataset();
        JFreeChart lineChart = ChartFactory.createXYLineChart("Training data",
                "Epoch",
                "Weights",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        ChartPanel chartPanel = new ChartPanel(lineChart);
        chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
        setContentPane(chartPanel);

        pack();
        RefineryUtilities.centerFrameOnScreen(this);
        setVisible(true);
    }

    public void addPoint(int epoch, double loss, String label) {
        addPoint(epoch, loss, label, true);
    }

    public void addPoint(int epoch, double loss, String label, boolean update) {
        dataset.add(epoch,loss, label, update);
    }
}