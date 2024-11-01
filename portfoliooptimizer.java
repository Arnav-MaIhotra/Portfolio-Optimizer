import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

class StockData {
    private String date;
    private double close;

    public StockData(String date, double close) {
        this.date = date;
        this.close = close;
    }

    public String getDate() {
        return date;
    }

    public double getClose() {
        return close;
    }

    @Override
    public String toString() {
        return "Date: " + date + ", Close: " + close;
    }
}

public class portfoliooptimizer {
    
    private static double[] time;
    private static double[][] returns;
    private static final String DATE_FORMAT = "MM/dd/yyyy HH:mm:ss";
    private static final String REFERENCE_DATE_STRING = "01/01/1970 00:00:00";
    public static void main(String[] args) {

        String path1 = "C:\\Users\\arnav\\OneDrive\\Documents\\Stock1.csv";
        String path2 = "C:\\Users\\arnav\\OneDrive\\Documents\\Stock2.csv";
        String path3 = "C:\\Users\\arnav\\OneDrive\\Documents\\Stock3.csv";
        String path4 = "C:\\Users\\arnav\\OneDrive\\Documents\\Stock4.csv";
        String path5 = "C:\\Users\\arnav\\OneDrive\\Documents\\Stock5.csv";
        String path6 = "C:\\Users\\arnav\\OneDrive\\Documents\\Stock6.csv";
        String path7 = "C:\\Users\\arnav\\OneDrive\\Documents\\Stock7.csv";

        List<StockData> Stock1 = readCSV(path1);
        List<StockData> Stock2 = readCSV(path2);
        List<StockData> Stock3 = readCSV(path3);
        List<StockData> Stock4 = readCSV(path4);
        List<StockData> Stock5 = readCSV(path5);
        List<StockData> Stock6 = readCSV(path6);
        List<StockData> Stock7 = readCSV(path7);

        List<List<StockData>> fullStockData = Arrays.asList(Stock1, Stock2, Stock3, Stock4, Stock5, Stock6, Stock7);
        
        double[][] covarianceMatrix = calculateCovarianceMatrix(fullStockData);

        time = toDays(Stock1);

        returns = transpose(calculateReturns(fullStockData));

        double[][] expectedReturns = new double[returns.length][2];
        double[] meanExpectedReturns = new double[returns[0].length];

        for (int i = 0; i < returns.length; i++) {
            expectedReturns[i] = calculateParameters(returns[i]);
            meanExpectedReturns[i] = expectedReturns[i][0];
        }

        double[] optimizedWeights = optimizePortfolio(0.1, covarianceMatrix, meanExpectedReturns);

        for (double i : optimizedWeights) {
            System.out.println(i);
        }

        double initialPrice = 100.0;
        double expectedReturn = 0.05;
        double volatility = 0.2;
        double time = 2;
        int numSims = 100000;

        double[] finalPrices = simulateFinalPrices(initialPrice, expectedReturn, volatility, time, numSims);
        
        double avgPrice = calculateAverage(finalPrices);
        double stdDev = calculateStandardDeviation(finalPrices, avgPrice);
        
        double var = calculateVaR(finalPrices, initialPrice, 0.05);
        double expectedShortfall = calculateExpectedShortfall(finalPrices, initialPrice, 0.05);
        
        System.out.println("Expected price in " + Double.toString(time).replaceAll((String)"\\.0$", "") + " years: $" + avgPrice);
        System.out.println("Standard deviation = " + stdDev);
        System.out.println("VaR = " + var);
        System.out.println("Expected Shortfall = " + expectedShortfall);
    }

    private static double[] toDays(List<StockData> stockDataList) {
        double[] days = new double[stockDataList.size()];
        try {
            Date referenceDate = new SimpleDateFormat(DATE_FORMAT).parse(REFERENCE_DATE_STRING);
            for (int i = 0; i < stockDataList.size(); i++) {
                String dateString = stockDataList.get(i).getDate();
                Date currentDate = new SimpleDateFormat(DATE_FORMAT).parse(dateString);
                long diffInMillis = currentDate.getTime() - referenceDate.getTime();
                long diffInDays = diffInMillis / (1000 * 60 * 60 * 24);
                days[i] = (double) diffInDays;
            }
        } catch (ParseException e) {
            e.printStackTrace();
        }
        return days;
    }

    private static double[] calculateParameters(double[] rets) {
        int n = rets.length;

        double[][] X = new double[n][2];
        double[] y = new double[n];

        for (int i = 0; i < n; i++) {
            X[i][0] = 1;
            X[i][1] = time[i];
            y[i] = rets[i];
        }

        double[][] XTX = matrixMultiply(transpose(X), X);
        
        double[] XTy = matrixMultiply(transpose(X), y);

        double[][] XTXInv = invertMatrix(XTX);

        double[] theta = matrixMultiply(XTXInv, XTy);
        double alpha = theta[0];
        double beta = theta[1];

        double[] params = {alpha, beta};

        return params;
    }

    private static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    private static double[] matrixMultiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            result[i] = 0;
            for (int j = 0; j < matrix[0].length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    private static double[][] matrixMultiply(double[][] matrix1, double[][] matrix2) {
        int rows = matrix1.length;
        int cols = matrix2[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = 0;
                for (int k = 0; k < matrix1[0].length; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return result;
    }

    private static double[][] invertMatrix(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][i + n] = 1;
        }

        for (int i = 0; i < n; i++) {
            double diag = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= diag;
            }

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    double factor = augmented[j][i];
                    for (int k = 0; k < 2 * n; k++) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }

        double[][] inverted = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverted[i][j] = augmented[i][j + n];
            }
        }
        return inverted;
    }

    public static double predict(double timeValue, double[] params) {
        return params[0] + params[1] * timeValue;
    }

    private static double[][] calculateReturns(List<List<StockData>> stocks) {
        int numDays = stocks.get(0).size();
        int numStocks = stocks.size();
        double[][] returns = new double[numDays - 1][numStocks];

        for (int i = 1; i < numDays; i++) {
            for (int j = 0; j < numStocks; j++) {
                double currentPrice = stocks.get(j).get(i).getClose();
                double previousPrice = stocks.get(j).get(i - 1).getClose();
                returns[i - 1][j] = (currentPrice - previousPrice) / previousPrice;
            }
        }
        return returns;
    }

    private static double[][] calculateCovarianceMatrix(List<List<StockData>> stocks) {
        double[][] returns = calculateReturns(stocks);
        int numStocks = returns[0].length;
        double[][] covarianceMatrix = new double[numStocks][numStocks];

        for (int i = 0; i < numStocks; i++) {
            for (int j = i; j < numStocks; j++) {
                double covariance = calculateCovariance(returns, i, j);
                covarianceMatrix[i][j] = covariance;
                covarianceMatrix[j][i] = covariance;
            }
        }
        return covarianceMatrix;
    }

    private static double calculateCovariance(double[][] returns, int stock1, int stock2) {
        double mean1 = calculateMean(returns, stock1);
        double mean2 = calculateMean(returns, stock2);
        double covariance = 0.0;

        for (double[] dailyReturns : returns) {
            covariance += (dailyReturns[stock1] - mean1) * (dailyReturns[stock2] - mean2);
        }
        return covariance / (returns.length - 1);
    }

    private static double calculateMean(double[][] returns, int stock) {
        double sum = 0.0;
        for (double[] dailyReturns : returns) {
            sum += dailyReturns[stock];
        }
        return sum / returns.length;
    }

    private static double[] simulateFinalPrices(double initialPrice, double expectedReturn, double volatility, double time, int numSims) {
        double[] finalPrices = new double[numSims];
        Random random = new Random();

        for (int i = 0; i < numSims; i++) {
            double price = initialPrice;
            for (int t = 0; t < 252 * time; t++) {
                double rand = random.nextGaussian();
                price *= Math.exp((expectedReturn - 0.5 * volatility * volatility) * (1.0 / 252) + volatility * Math.sqrt(1.0 / 252) * rand);
            }
            finalPrices[i] = price;
        }
        return finalPrices;
    }

    private static double calculateAverage(double[] prices) {
        double sum = 0.0;
        for (double price : prices) {
            sum += price;
        }
        return sum / prices.length;
    }

    private static double calculateStandardDeviation(double[] prices, double avgPrice) {
        double sumSquaredDiffs = 0.0;
        for (double price : prices) {
            sumSquaredDiffs += Math.pow(price - avgPrice, 2);
        }
        return Math.sqrt(sumSquaredDiffs / prices.length);
    }

    private static double calculateVaR(double[] prices, double initialPrice, double percentile) {
        Arrays.sort(prices);
        int varIndex = (int) (percentile * prices.length);
        return initialPrice - prices[varIndex];
    }

    private static double calculateExpectedShortfall(double[] prices, double initialPrice, double percentile) {
        Arrays.sort(prices);
        int varIndex = (int) (percentile * prices.length);
        double sum = 0.0;
        
        for (int i = 0; i < varIndex; i++) {
            sum += prices[i];
        }
        
        return initialPrice - (sum / varIndex);
    }

    private static List<StockData> readCSV(String path) {
        String line;
        List<StockData> stockData = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine();
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                String date = values[0].trim();
                double close = Double.parseDouble(values[1].trim());
                stockData.add(new StockData(date, close));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return stockData;
    }

    private static double[] optimizePortfolio(double targetReturn, double[][] covarianceMatrix, double[] expectedReturns) {
        int numStocks = covarianceMatrix.length;
        double[] weights = new double[numStocks];
        double learningRate = 0.01;
        double tolerance = 1e-6;
        double currentReturn;
        double currentRisk;

        Arrays.fill(weights, 1.0/numStocks);

        boolean convergence = false;

        int iters = 0;

        while (!convergence && iters<100000000) {
            currentReturn = portfolioReturn(weights, expectedReturns);
            currentRisk = portfolioVariance(weights, covarianceMatrix);

            if (Math.abs(currentReturn - targetReturn) < tolerance) {
                convergence = true;
                break;
            }

            for (int i = 0; i < numStocks; i++) {
                weights[i] += learningRate * (currentReturn - targetReturn) * (expectedReturns[i] / currentRisk);
            }

            normalizeWeights(weights);
            iters++;
        }

        return weights;
    }

    private static double portfolioReturn(double[] weights, double[] expectedReturns) {
        double portfolioReturn = 0.0;
        for (int i = 0; i < weights.length; i++) {
            portfolioReturn += weights[i] * expectedReturns[i];
        }
        return portfolioReturn;
    }

    private static double portfolioVariance(double[] weights, double[][] covarianceMatrix) {
        double variance = 0.0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights.length; j++) {
                variance += weights[i] * weights[j] * covarianceMatrix[i][j];
            }
        }
        return variance;
    }

    private static void normalizeWeights(double[] weights) {
        double sum = Arrays.stream(weights).sum();
        for (int i = 0; i < weights.length; i++) {
            weights[i] /= sum;
        }
    }
}