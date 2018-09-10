#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "supervised.h"
#include "error.h"
#include "string.h"
#include "rand.h"
#include "vec.h"

using std::vector;

// virtual
void SupervisedLearner::filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out)
{
	feat_out.copy(feat_in);
	lab_out.copy(lab_in);
}

size_t SupervisedLearner::countMisclassifications(const Matrix& features, const Matrix& labels)
{
    if(features.rows() != labels.rows())
        throw Ex("Mismatching number of rows");
    size_t mis = 0;
    for(size_t i = 0; i < features.rows(); i++)
    {
        const Vec& pred = predict(features[i]);
        const Vec& lab = labels[i];
        for(size_t j = 0; j < lab.size(); j++)
        {
            if(pred[j] != lab[j])
            {
                mis++;
            }
        }
    }
    return mis;
}

// virtual
void SupervisedLearner::trainIncremental(const Vec& feat, const Vec& lab)
{
	throw Ex("Sorry, this learner does not support incremental training");
}

double SupervisedLearner::sumSquaredError(const Matrix& features, const Matrix& labels)
{
	
	double sse = 0.0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		const Vec& pred = predict(features[i]);
		const Vec& lab = labels[i];

		for(size_t j = 0; j < lab.size(); j++)
		{
			double d = lab[j] -  pred[j]; 
			sse += d*d;
		}
        
	}
    sse = sse/features.rows();

	return sse;
}

double SupervisedLearner::NFoldCrossVal(Matrix& features,  Matrix& labels, int folds, int iter){
	int frow = features.rows() / folds;
	int fcol = features.cols() ;
	int lrow = labels.rows() / folds;
	int lcol = labels.cols();
    Matrix feaTrain(frow*(folds-1),fcol);
    Matrix labTrain(lrow*(folds-1),lcol);
	Matrix feaTest(frow,fcol);
    Matrix labTest(lrow,lcol);
    //swapRows
    Rand ra(1);
    int a,b;
	double sse = 0.0;
	double rmse = 0.0;
    
    
    for(int j=0; j<5;j++){
        for(int k = features.rows()-1; k > 0; k--){
            a = k;
            b = rand()%features.rows();
            //std::cout<<"b:"<<b<<std::endl;
            features.swapRows(a,b);
            labels.swapRows(a,b);
        }
        for (int select = 0; select != folds; select++){
            feaTest.copyBlock(0,0,features,frow*select,0,frow,fcol);
            labTest.copyBlock(0,0,labels,lrow*select,0,lrow,lcol);
            
            if(select == 0){
                //void copyBlock(size_t destRow, size_t destCol, const Matrix& that, size_t rowBegin, size_t colBegin, size_t rowCount, size_t colCount);
				feaTrain.copyBlock(0,0,features,frow,0,frow*(folds-1),fcol);
				labTrain.copyBlock(0,0,labels,lrow,0,lrow*(folds-1),lcol);
                
				train(feaTrain,labTrain);
				sse += sumSquaredError(feaTest,labTest);
            }
            else{
                feaTrain.copyBlock(0,0,features,0,0,frow*select,fcol);
                labTrain.copyBlock(0,0,labels,0,0,lrow*select,lcol);
                
                feaTrain.copyBlock(frow*select,0,features,frow*(select+1),0,frow*(folds-select-1),fcol);
                labTrain.copyBlock(lrow*select,0,labels,lrow*(select+1),0,lrow*(folds-select-1),lcol);
                
                train(feaTrain,labTrain);
                sse += sumSquaredError(feaTest,labTest);
            }
		}
	}
    
	rmse = (double)sqrt((double)sse / (folds * 5)) ;

	return rmse;	
}






