package pro4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.Vector;

/**
 * This program implements a k-nearest neighbor (kNN) classifer. The program
 * should take three arguments as input: the value of k (an integer > 0), the
 * name of the training data file, and the name of the testing data file.
 *
 *
 * @author daisy
 */
public class knn {

    private int row_te, row_tr, column_te, column_tr, kN;
    Matrix train, test, train_x;
    Vector kmean;

    public knn() {
        row_tr = row_te = 1;
        column_tr = column_te = 1;
    }

    private void readTrainingData(String file) {

        //File will be read into a matrix.
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));

            //Read the first line to get the Dimension.
            String line = br.readLine();
            line = line.trim();

            //Split function is used to split the content of line by space.
            String[] splited = line.split("\\s+");

            //Set the matrix M and N.
            row_tr = Integer.parseInt(splited[0]);
            column_tr = Integer.parseInt(splited[1]);
            train = new Basic2DMatrix(new double[row_tr][column_tr + 1]);
            train_x = new Basic2DMatrix(new double[row_tr][column_tr]);

            //Make the file into a matrix.
            for (int i = 0; i < row_tr; i++) {
                line = br.readLine();
                if (line == null) {
                    break;
                } else {
                    line = line.trim();
                    splited = line.split("\\s+");

                    for (int j = 0; j < column_tr + 1; j++) {
                        train.set(i, j, Double.parseDouble(splited[j]));
                        if (j < column_tr) {
                            train_x.set(i, j, Double.parseDouble(splited[j]));
                        }
                    }
                }
            }

            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void readTestingData(String k, String file) {
        //Set how many neighbour it will return.
        kN = Integer.parseInt(k);

        Vector intermediate;

        //File will be read into a matrix.
        try {

            BufferedReader br = new BufferedReader(new FileReader(file));

            //Read the first line to get the Dimension.
            String line = br.readLine();
            line = line.trim();

            //Split function is used to split the content of line by space.
            String[] splited = line.split("\\s+");

            //Set the matrix M and N.
            row_te = Integer.parseInt(splited[0]);
            column_te = Integer.parseInt(splited[1]);
            test = new Basic2DMatrix(new double[row_te][column_te]);

            //Make the file into a matrix.
            for (int i = 0; i < row_te; i++) {
                line = br.readLine();
                if (line == null) {
                    break;
                } else {
                    line = line.trim();
                    splited = line.split("\\s+");

                    for (int j = 0; j < column_te; j++) {
                        test.set(i, j, Double.parseDouble(splited[j]));
                    }
                }
            }

            Comparator<Pair> cmp = new PairComparator();
            Comparator<Label> cmp1 = new LabelComparator();

            //Compute the distance and return the corresponding labels.
            for (int i = 0; i < row_te; i++) {
                List<Pair> list = new ArrayList<Pair>();

                for (int j = 0; j < row_tr; j++) {

                    intermediate = test.getRow(i).subtract(train_x.getRow(j));
                    double distance = intermediate.norm();

                    list.add(new Pair(distance, i, j, (int) (train.get(j, column_tr))));
                }

                Collections.sort(list, cmp);

                //This is the list of labels to be compute.
                List<Label> lbl = new ArrayList<>();

                for (int m = 0; m < kN; m++) {

                    if (!lbl.isEmpty()) {
                        boolean needAdd = true;
                        for (int x = 0; x < lbl.size(); x++) {
                            if (list.get(m).label == lbl.get(x).label) {
                                lbl.get(x).setCount(lbl.get(x).count + 1);
                                needAdd = false;
                                break;
                            }
                        }
                        if (needAdd) {
                            lbl.add(new Label(list.get(m).label, list.get(m)));
                        }
                    } else {
                        lbl.add(new Label(list.get(m).label, list.get(m)));
                    }
                }

                Collections.sort(lbl, cmp1);

                System.out.println(i + 1 + ". " + test.getRow(i) + " -- " + lbl.get(0).label);
            }

            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void run(String k, String trainingFile, String testingFile) {
        readTrainingData(trainingFile);
        readTestingData(k, testingFile);

    }

    public class PairComparator implements Comparator<Pair> {

        @Override
        public int compare(Pair o2, Pair o1) {
            if (o2.output - o1.output > 0) {
                return 1;
            } else if (o2.output - o1.output < 0) {
                return -1;
            } else {
                return 0;
            }
        }
    }

    public class LabelComparator implements Comparator<Label> {

        @Override
        public int compare(Label l1, Label l2) {
            if (l1.count - l2.count > 0) {
                return -1;
            } else if (l1.count - l2.count < 0) {
                return 1;
            } else {
                if (l1.distance < l2.distance) {
                    return -1;
                } else if (l1.distance < l2.distance) {
                    return 1;
                }
                return 0;
            }
        }
    }

    public class Label {

        int label;

        public int getCount() {
            return count;
        }

        public void setCount(int count) {
            this.count = count;
        }
        int count;
        double distance;

        public Label() {
        }

        public Label(Label l) {
            this.label = l.label;
            this.count = l.count;
            this.distance = l.distance;
        }

        public Label(int i, Pair p) {
            label = i;
            count = 1;
            distance = p.output;
        }

        @Override
        public String toString() {
            return "L: " + label + " Count: " + count;
        }

    }

    public class Pair {

        double output;
        int index_train_x;
        int index_test;
        int label;

        public Pair() {
        }

        public Pair(double distance, int test_row, int train_row, int l) {
            output = distance;
            index_test = test_row;
            index_train_x = train_row;
            label = l;
        }

        @Override
        public String toString() {
            return "Distance: " + output + ", index_train_x=" + index_train_x + ", index_test=" + index_test + ", label=" + label + "\n";
        }
    }

    public static void main(String args[]) {
        knn n = new knn();
        n.run(args[0], args[1], args[2]);
    }
}
