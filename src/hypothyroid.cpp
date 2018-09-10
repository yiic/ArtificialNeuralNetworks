// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include <iomanip>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "baseline.h"
#include "preprocess.h"
#include "imputer.h"
#include "nomcat.h"
#include "normalizer.h"
#include "svg.h"
#include <cmath>
#include <algorithm>

#ifdef WINDOWS
#    include <windows.h>
#else
#    include <sys/time.h>
#endif

double seconds()
{
#ifdef WINDOWS
    return (double)GetTickCount() * 1e-3;
#else
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
#endif
}

using namespace std;


class Layer
{
public: Vec activation;
public: Vec blame;
public: int weights_dim;
    
public:
    Layer(size_t inputs, size_t outputs) :
    activation(outputs),
    blame(outputs){
        weights_dim = outputs + inputs * outputs;
    }
    virtual ~Layer(){
    }
    
    // Basic functions
    virtual void activate( Vec& weights, const Vec& x) = 0;
    virtual void backprop( Vec& weights, Vec& prevBlame) = 0;
    virtual void update_gradient(const Vec& x, Vec& gradient) = 0;
};

//============================================================
// LayerLinear
//============================================================
class LayLinear : public Layer
{
public:
    LayLinear (size_t inputs, size_t outputs) :
    Layer(inputs,outputs){
    }
    
    virtual ~LayLinear(){
    }
    
    // ---------------------------------------------------
    // Activate
    virtual void activate( Vec& weights, const Vec& x){
        
        VecWrapper b(weights, 0, weights.size() / (1 + x.size()));
        VecWrapper M(weights, b.size(), weights.size());
        
        for(size_t i = 0; i < b.size(); i++){
            VecWrapper m(M, x.size()*i, x.size());
            activation[i] = m.dotProduct(x);
        }
        
        activation += b;
    }
    
    // ---------------------------------------------------
    // Backpropagation -> prevBlame
    virtual void backprop( Vec& weights, Vec& prevBlame){
        
        int output_size = blame.size();
        int input_size = prevBlame.size();
        
        VecWrapper b(weights, 0, blame.size());
        VecWrapper M(weights, b.size(), input_size * output_size);
        
        // prevBlame = M^T * blame
        Vec mt_row(output_size);
        for(int i = 0; i < input_size; i++){
            for(int j = 0; j < output_size; j++)
                mt_row[j] = M[j * input_size + i];
            
            prevBlame[i] = mt_row.dotProduct(blame);
        }
    }
    
    // ---------------------------------------------------
    // Update Gradient
    virtual void update_gradient( const Vec& x, Vec& gradient){
        VecWrapper gb(gradient, 0, blame.size());
        VecWrapper gM(gradient, gb.size(), gradient.size());
        
        // update b
        gb += blame;
        
        // update M
        Matrix temp(outer_product(blame, x));
        for(size_t i = 0; i < temp.rows(); i++){
            for (size_t j = 0; j < temp.cols(); j++){
                gM[i * temp.cols() + j] += temp[i][j];
            }
        }
        
    }
    
    // ---------------------------------------------------
    // Ordinary_least_squares: compute M and b by knowing y and x
    void ordinary_least_squares( const Matrix& X,  const Matrix& Y, Vec& weights){
        
        Vec x_centroid(X.cols());
        Vec y_centroid(Y.cols());
        
        Matrix origin_centered_x(0,X.cols());
        Matrix origin_centered_y(0,Y.cols());
        
        for(size_t i = 0; i < X.cols(); i++)
            x_centroid.data()[i] = X.columnMean(i);
        for(size_t i = 0; i < Y.cols(); i++)
            y_centroid.data()[i] = Y.columnMean(i);
        
        // origin centered data
        for(size_t i = 0; i < X.rows(); i++){
            origin_centered_x.newRow().copy(X.crow(i) - x_centroid);
        }
        
        for(size_t i = 0; i < Y.rows(); i++){
            origin_centered_y.newRow().copy(Y.crow(i) - y_centroid);
        }
        
        // calculate M
        Matrix yx(origin_centered_y.cols(), origin_centered_x.cols());
        yx.fill(0.0);
        Matrix xx(origin_centered_x.cols(), origin_centered_x.cols());
        xx.fill(0.0);
        
        for(size_t i = 0; i < origin_centered_y.rows(); i++){
            Matrix temp_yx(outer_product(origin_centered_x.row(i), origin_centered_y.row(i)));
            for(size_t k = 0; k < origin_centered_y.cols(); k++)
                yx.row(k) += temp_yx.row(k);
        }
        
        for(size_t i = 0; i < origin_centered_x.rows(); i++){
            Matrix temp_xx = outer_product(origin_centered_x.row(i), origin_centered_x.row(i));
            for(size_t k = 0; k < origin_centered_x.cols(); k++)
                xx.row(k) += temp_xx.row(k);
        }
        
        Matrix *xx_inverse = xx.pseudoInverse();
        Matrix *M = xx_inverse->multiply(yx, *xx_inverse,false,false);
        
        // calculate b
        Vec b(M->rows());
        for(size_t i = 0; i < M->rows(); i++){
            b.data()[i]= M->row(i).dotProduct(x_centroid);
        }
        b.copy(y_centroid  - b);
        
        //add b to weights
        weights.resize(b.size() + M->rows()*M->cols());
        for(size_t i = 0; i < b.size(); i++){
            weights.data()[i] = b.data()[i];
        }
        
        //add M to weights
        for(size_t i = 0; i < M->rows(); i++){
            for(size_t j = 0; j < M->cols(); j++){
                weights.data()[b.size()+i*M->cols()+j] = M->row(i).data()[j];
            }
        }
        
    }
    
    // ---------------------------------------------------
    // Outer Product of two Vec's : a * b
    Matrix outer_product(Vec& a, Vec& b){
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++){
            for (size_t j = 0; j < b.size(); j++)
                result[i][j] = a[i] * b[j];
        }
        return result;
    }
    
    Matrix outer_product(const Vec& a,  Vec& b){
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++){
            for (size_t j = 0; j < b.size(); j++)
                result[i][j] = a[i] * b[j];
        }
        return result;
    }
    
    Matrix outer_product(Vec& a, const Vec& b){
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++){
            for (size_t j = 0; j < b.size(); j++)
                result[i][j] = a[i] * b[j];
        }
        return result;
    }
    
    
};

//============================================================
// LayerTanh
//============================================================
class LayerTanh : public Layer
{
public:
    LayerTanh (size_t inputs) :
    Layer(inputs,inputs){
        weights_dim = 0;
    }
    
    virtual ~LayerTanh(){
    }
    // ---------------------------------------------------
    // Activate
    virtual void activate(Vec& weights, const Vec& x){
        for(size_t i = 0; i < x.size(); i++){
            activation[i] = tanh(x[i]);
        }
    }
    // ---------------------------------------------------
    // Backpropagation
    virtual void backprop( Vec& weights, Vec& prevBlame){
        for(size_t i = 0; i < prevBlame.size(); i++){
            prevBlame[i] = blame[i] * (1.0 - (activation[i] * activation[i]));
        }
    }
    
    // ---------------------------------------------------
    // Update Gradient
    virtual void update_gradient( const Vec& x, Vec& gradient){
        
    }
    
};

//============================================================
// NeuralNet
//============================================================
class NeuralNet : public SupervisedLearner
{
public:
    Vec weights;
    //Vec gradients;
    Vec act;
    
    vector<int> net_info;
    vector<Layer*> layers;
    
    // Constructors
    NeuralNet(vector<int> net_sets): weights(1){
        net_info = net_sets;
        
        // set layers
        int weights_len = 0;
        for (int i = 0; i < net_sets.size() - 1; i++){
            if(i % 2 == 0){
                layers.push_back(new LayLinear(net_sets[i], net_sets[i + 1]));
                weights_len += net_sets[i + 1] + net_sets[i + 1] * net_sets[i];
            }else{
                layers.push_back(new LayerTanh(net_sets[i + 1]));
            }
        }
        
        weights.resize(weights_len);
    }
    
    NeuralNet(size_t w): weights(w){ }
    NeuralNet(){ }
    
    virtual ~NeuralNet(){
        for(size_t i = 0; i < layers.size(); i++)
            delete(layers[i]);
    }
    
    //----------------------------------------------------------
    virtual const char* name() { return "NeuralNet"; }
    
    //----------------------------------------------------------
    //// Hw1: Train
    virtual void train(const Matrix& features, const Matrix& labels){
        LayLinear tLayer(1, 1);
        tLayer.ordinary_least_squares(features, labels, weights);
    }
    
    //----------------------------------------------------------
    // Initial Weights
    virtual void initWeights(){
        Rand rand(2);
        int pos = 0;
        int input_dim;
        
        for(int i = 0; i < layers.size(); i+=2){
            VecWrapper layer_weights(weights, pos, layers[i]->weights_dim);
            pos += layers[i]->weights_dim;
            
            input_dim = layers[i]->weights_dim / layers[i]->blame.size() - 1;
            
            for(int j = 0; j < layer_weights.size(); j++){
                layer_weights[j] = max(0.03, 1.0 / input_dim) * rand.normal();
            }
        }
    }
    
    //----------------------------------------------------------
    // Predict -> activation
    virtual const Vec& predict(const Vec& net_input){
        int pos = 0;
        
        // activate all layers
        for(int i = 0; i < layers.size(); i++){
            VecWrapper layer_weights(weights, pos, layers[i]->weights_dim);
            pos += layers[i]->weights_dim;
            
            if(i == 0){
                layers[i]->activate(layer_weights, net_input);
            }else{
                layers[i]->activate(layer_weights, layers[i - 1]->activation);
            }
        }
        
        act.copy(layers[layers.size() - 1]->activation);
        
        return act;
    }
    
    //----------------------------------------------------------
    // Backprop
    virtual void backProp(const Vec targets){
        
        // output layer blame
        layers[layers.size() - 1]->blame.copy(targets - layers[layers.size() - 1]->activation);
        
        // back propagation
        int pos = 0;
        for(int i = layers.size() - 1; i > 0; i--){
            pos += layers[i]->weights_dim;
            VecWrapper layer_weights(weights, weights.size() - pos, layers[i]->weights_dim);
            
            layers[i]->backprop(layer_weights, layers[i - 1]->blame);
        }
    }
    
    //----------------------------------------------------------
    // Update Gradient
    virtual void updateGradient(const Vec& net_input, Vec& gradients){
        int pos = 0;
        
        for(int i = 0; i < layers.size(); i++){
            VecWrapper layer_gradients(gradients, pos, layers[i]->weights_dim);
            pos += layers[i]->weights_dim;
            
            if (i == 0)
                layers[i]->update_gradient(net_input, layer_gradients);
            else
                layers[i]->update_gradient(layers[i - 1]->activation, layer_gradients);
        }
        
    }
    
    //----------------------------------------------------------
    // Refine Weights (gradient decent)
    virtual void refineWeights(const Vec& net_input, const Vec& targets, double lr, int batch_size){
    
        Vec gradients(weights.size());
        
//        // Momentum Stochastic
//        // need to decrease lr: HW3: batch_size = 4, lr = 0.001 with decay=0.6)
//        double momentum = 1.0 - 1.0 / batch_size;
//
//        predict(net_input);
//        backProp(targets);
//
//        gradients.copy(gradients * momentum);
//        updateGradient(net_input, gradients);
//
//        weights += gradients * lr;
        

        
        // Stochastic (MNIST: lr = 0.03; HW3: lr = 0.003 with decay=0.8)
        predict(net_input);
        backProp(targets);
        gradients.fill(0.0);
        updateGradient(net_input, gradients);

        weights += gradients * lr;
        
        
        
        
        // (Mini) Batch (MINIST: minibatch<=40 or decrease the lr;  HW3: lr = 0.003)
//        int minibatch = batch_size;
//        static int t = 0;
//
//        if (t < minibatch){
//            predict(net_input);
//            backProp(targets);
//            updateGradient(net_input, gradients);
//            t++;
//
//        }else{
//            weights += gradients * lr;
//            t = 0;
//            gradients.fill(0.0);
//
//        }
        
        ////cout << "updated weights: "; weights.print(); cout << endl;
        
    }
    
    
    
};

//=============================================================
// Unit tests
//=============================================================

// "debug spew" test
void h2_mini_test(){
    vector<int> net_sets = {1, 2, 2, 1};
    Vec feat_vec({0.3});
    Vec tar_vec({0.7});
    
    NeuralNet test_net(net_sets);
    test_net.weights.copy({0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3});
    
    //    test_net.refineWeights(feat_vec, tar_vec, 0.1);
    //    test_net.refineWeights(feat_vec, tar_vec, 0.1);
    //    test_net.refineWeights(feat_vec, tar_vec, 0.1);
    
}

//=======================================================================
// Filter (Data Prepocess)
//=======================================================================
class Filter{
    
public: NomCat nomcat;
public: Normalizer normal;
public: Imputer inputer;
public: Matrix after;
    
public:
    Filter(Matrix& before, Matrix& after){
        Matrix middle1, middle2;
        
        //Imputer
        inputer.train(before);
        middle1.setSize(before.rows(), inputer.outputTemplate().cols());
        for(int i = 0; i < before.rows(); i++){
            inputer.transform(before[i], middle1[i]);
        }
    
        //Nomcat
        nomcat.train(before);
        middle2.setSize(before.rows(), nomcat.outputTemplate().cols());
        for(int i = 0; i < before.rows(); i++){
                nomcat.transform(middle1[i], middle2[i]);
        }
        
        //Normalizer
        normal.train(middle2);
        after.setSize(before.rows(), normal.outputTemplate().cols());
        for(int i = 0; i < before.rows(); i++){
                normal.transform(middle2[i], after[i]);
        }
        //cout << "Data prepocess is done." <<endl;
    }

    virtual ~Filter(){}
    
    // Untransform a Vec of pattern data
    Vec untran(Vec in){
        Vec mid;
        Vec out;
        normal.untransform(in, mid);
        nomcat.untransform(mid, out);
        return out;
    }

};


//=======================================================================
// MNIST test
//=======================================================================
int MNIST_test(NeuralNet& trained_model, Matrix& test_features, Matrix& testLabels, Filter& fil_test_lab){

    if(test_features.rows() != testLabels.rows())
        throw Ex("Mismatching number of rows");
    
    int predict;
    int target;
    int mis = 0;
    
    for(int i = 0; i < test_features.rows(); i++){
        
        predict = fil_test_lab.untran(trained_model.predict(test_features[i]))[0];
        target = testLabels[i][0];
        
        if(predict != target)
            mis++;
    }
    
    //cout << "misclassification: " << mis << endl;
    
    return mis;
}


//=======================================================================
// Train
//=======================================================================
void Train(int batch_size){
    
    //Load the data
    string fn = "../bin/data/";
    Matrix allData;
    allData.loadARFF(fn + "hypothyroid.arff");
    
    
    //divide into training and testing data
    int total_num = allData.rows();
    int train_num = total_num * 0.7;
    int test_num = total_num - train_num;
    
    Matrix trainFeatures(train_num, 29), trainLabels(train_num, 1), testFeatures(test_num, 29), testLabels(test_num, 1);
    trainFeatures.copyBlock(0, 0, allData, 0, 0, train_num, 29);
    trainLabels.copyBlock(0, 0, allData, 0, 29, train_num, 1);
    testFeatures.copyBlock(0, 0, allData, train_num, 0, test_num, 29);
    testLabels.copyBlock(0, 0, allData, train_num, 29, test_num, 1);
    
    // Data preprocess
    Matrix train_features;
    Matrix train_labels;
    Matrix test_features;
    Matrix test_labels;
    Filter fil_train_fea(trainFeatures, train_features);
    Filter fil_train_lab(trainLabels, train_labels);
    Filter fil_test_fea(testFeatures, test_features);
    Filter fil_test_lab(testLabels, test_labels);
    
    
    // Initailize the NeuralNet
    vector<int> net_sets = {(int)train_features.cols(), 80, 80, (int)train_labels.cols(), (int)train_labels.cols()};
    NeuralNet bp_learner(net_sets);
    bp_learner.initWeights();
    
    //cout << "before training: ";

    int mis = MNIST_test(bp_learner, test_features, testLabels, fil_test_lab);
    double mispersentage = mis / test_features.rows();
    int last_ep_mis = mis;
    
    // TRAIN begin
    double bg_time = seconds();
    int epoch = 0;
    bool convergence = false;
    
     while(!convergence){
        if(train_features.rows() != train_labels.rows())
            throw Ex("Mismatching number of rows");
        else{
            // create an array for row index
            vector<int> select_index(train_features.rows());
            int temp = 0;
            for(int i = 0; i < select_index.size(); i++){
                select_index[i] = i;
            }
            
            int check = 0;
            double lr = 0.001;
            
            // loop over every pattern randomly
            random_shuffle(select_index.begin(), select_index.end());
            for(vector<int>::iterator it = select_index.begin(); it != select_index.end(); ++it){
                
                // update weights
                bp_learner.refineWeights(train_features[*it], train_labels[*it], lr, batch_size);
                
                // output misclassification persentage
                if(check % 100 == 0){
                    double cur_time = seconds() - bg_time;

                    // testing data
                    mis = MNIST_test(bp_learner, test_features, testLabels, fil_test_lab);
                    mispersentage = mis * 1.0 / test_features.rows();
                    cout << cur_time <<", " << mispersentage << endl;
                    
                    // training data
//                    mis = MNIST_test(bp_learner, train_features, trainLabels, fil_train_lab);
//                    double mispersentage = mis * 1.0 / train_features.rows();
//                    cout << cur_time <<", " << mispersentage << endl;
                }
                
            // lr decay
                lr*=0.6;
                check++;
                
            }//end for loop
            
        }//end else
        
        epoch++;
        
        // detect convergence
        if(epoch % 4 == 0){
            cout << "convergence detecting... " << endl;
            
            int cur_ep_mis = MNIST_test(bp_learner, test_features, testLabels, fil_test_lab);
            
            if ((last_ep_mis - cur_ep_mis) * 1.0 / last_ep_mis < 0.02){
                convergence = true;
                cout << "convergence detected! " <<endl;
            }

            last_ep_mis = cur_ep_mis;

        }

    }//end while loop
    
}


//=======================================================================
int main(int argc, char *argv[])
{
    enableFloatingPointExceptions();
    
    int ret = 1;
    
    // Get mini_batch size from user
    int batch_size = 10;
    
    if (argc == 2){
        batch_size = atoi(argv[1]);
    }else{
        cout << "Enter the mini batch size: ";
        cin >> batch_size;
    }
    
    
    try{
        Train(batch_size);
        ret =0;
    }catch(const std::exception& e){
        cerr << "An error occurred: " << e.what() << "\n";
    }
    
    
    return ret;
}

