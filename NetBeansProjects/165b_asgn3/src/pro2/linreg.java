package pro2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.la4j.LinearAlgebra;
import org.la4j.inversion.MatrixInverter;
import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.Vector;
import org.la4j.vector.dense.BasicVector;

/**
 * The program should take two arguments as input: the name of the training data
 * file and the name of the testing data file. Results are output to the
 * standard output in the following format: Line 1: Regression values learned
 * from the training set (e.g., 2.5, -1.0, 1.0) Lines 2 through M+1: testing
 * data values (e.g., 3.00, 2.72, -12.24) and regression estimate (e.g., 2.55)
 * for each data point in the testing set.
 *
 * @author daisy
 */
public class linreg {

    private int row_te, row_tr, column_te, column_tr;
    Vector w;
    double t;

    public linreg() {
        row_tr = row_te = 1;
        column_tr = column_te = 1;
    }

    private void readTrainingData(String file) throws FileNotFoundException, IOException {
        //Declare variables.
        Matrix a;
        Matrix x = null;
        Vector one, y;

        try {
            ArrayList<String> strArray = new ArrayList<>();
            BufferedReader br = new BufferedReader(new FileReader(file));

            //Read the first line to get the Dimension.
            String line = br.readLine();
            line = line.trim();

            //Split function is used to split the content of line by space.
            String[] splited = line.split("\\s+");

            //Set the matrix M and N.
            row_tr = Integer.parseInt(splited[0]);
            column_tr = Integer.parseInt(splited[1]);
            a = new Basic2DMatrix(new double[row_tr][column_tr]);
            x = new Basic2DMatrix(new double[row_tr][column_tr]);
            one = new BasicVector(row_tr);
            y = new BasicVector(row_tr);

            //Initialize the t vector
            for (int i = 0; i < row_tr; i++) {
                one.set(i, 1);
            }

            //Read in the file into a String Array List.
            for (int j = 0; j < row_tr * column_tr; j++) {
                line = br.readLine();
                if (line == null) {
                    break;
                } else {
                    line = line.trim();
                    strArray.addAll(Arrays.asList(line.split("\\s+")));
                }
            }

            //Make the file into a matrix.
            int k = 0;
            for (int i = 0; i < row_tr; i++) {
                for (int j = 0; j < column_tr; j++) {
                    if (strArray.size() <= k) {
                        break;
                    } else {
                        a.set(i, j, Double.parseDouble(strArray.get(k)));
                        k++;
                    }
                }
            }

            //Set the X matrix.
            for (int i = 0; i < row_tr; i++) {
                for (int j = 0; j < column_tr - 1; j++) {
                    x.set(i, j, a.get(i, j));

                }
            }

            //Save another column in X matrix for t vector
            x.setColumn(column_tr - 1, one);

            //Set the y vector.
            for (int i = 0; i < row_tr; i++) {

                y.set(i, a.get(i, column_tr - 1));

            }

            //Calculating the w vector and the value of t.
            Matrix S = x.transpose().multiply(x);
            MatrixInverter inverter = S.withInverter(LinearAlgebra.GAUSS_JORDAN); //Generated automatically.
            Matrix inversed_S = inverter.inverse(LinearAlgebra.DENSE_FACTORY);//Inverse the matrix with a inverter.
            Vector wt = inversed_S.multiply(x.transpose().multiply(y));
            t = wt.get(0);

            //Initialize the w vector.
            w = new BasicVector(column_tr - 1);

            for (int i = 0; i < column_tr - 1; i++) {
                w.set(i, wt.get(i));
            }
            System.out.println("w, t = " + w + ", " + t);
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void readTestingData(String file) {
        //Declare variables.
        Matrix x;
        Vector y;

        try {
            ArrayList<String> strArray = new ArrayList<>();
            BufferedReader br = new BufferedReader(new FileReader(file));

            //Read the first line to get the Dimension.
            String line = br.readLine();
            line = line.trim();

            //Split function is used to split the content of line by space.
            String[] splited = line.split("\\s+");

            //Set the matrix M and N.
            row_te = Integer.parseInt(splited[0]);
            column_te = Integer.parseInt(splited[1]);
            x = new Basic2DMatrix(new double[row_te][column_te]);

            //Read in the file into a String Array List.
            for (int j = 0; j < row_te * column_te; j++) {
                line = br.readLine();
                if (line == null) {
                    break;
                } else {
                    line = line.trim();
                    strArray.addAll(Arrays.asList(line.split("\\s+")));
                }
            }

            //Make the file into a matrix.
            int k = 0;
            for (int i = 0; i < row_te; i++) {
                for (int j = 0; j < column_te; j++) {
                    if (strArray.size() <= k) {
                        break;
                    } else {
                        x.set(i, j, Double.parseDouble(strArray.get(k)));
                        k++;
                    }
                }
            }

            y = x.multiply(w);
            k = 1;
            for (int i = 0; i < row_te; i++) {

                System.out.println(k + ". " + x.getRow(i) + " -- " + y.get(i));
                k++;
            }
            br.close();
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[]) {
        linreg obj = new linreg();

        try {
            obj.readTrainingData(args[0]);
            obj.readTestingData(args[1]);

        } catch (IOException ex) {
            Logger.getLogger(linreg.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
    }
}
