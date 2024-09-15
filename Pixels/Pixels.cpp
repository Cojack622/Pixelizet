// Pixels.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/interface.h>

using namespace std; 
// 
//struct LAB_Color {
//    char l, a, b;
//};

class SuperPixel;
class PaletteColor;

//Structs
struct Pixel {
    cv::Vec3f color;
    cv::Point2f position;
};

struct Cluster {
    cv::Vec3f color;
    int sub1Index, sub2Index;
};


//Helper methods
double ColorDifference(cv::Vec3f color1, cv::Vec3f color2) {
    //cv::Vec3f diffVec= color2 - color1;
    double sum = 0.0;
    /*for (int i = 0; i < 3; i++) {
        sum += powf(diffVec[i], 2);
    }*/
    sum += powf(color2[0] - color1[0], 2);
    sum += powf(color2[1] - color1[1], 2);
    sum += powf(color2[2] - color1[2], 2);

    return sqrt(sum);
}

double PositionDifference(cv::Point2f point1, cv::Point2f point2){
    cv::Point2f diff = point2 - point1; 
    double sum = powf(diff.x, 2) + powf(diff.y, 2);
    return sqrt(sum);
}


bool inBounds(cv::Point2i point, int w, int h) {
    return (point.x >= 0 && point.x < w && point.y >= 0 && point.y < h);
}

bool inBounds(int x, int y, int w, int h) {
    return (x >= 0 && x < w && y >= 0 && y < h);
}



//Classes

class SuperPixel {

    public:

        /*static double diffCoeff;*/

        cv::Vec3f averageColor;
        cv::Vec3f paletteColor;
        vector<Pixel> associatedPixels;
        vector<double> probToCluster;

        
        cv::Point2f position;

        void addToAvg(double l, double a, double b) {
            lAvg += l;
            aAvg += a;
            bAvg += b;
        }

        void setAvg(/*int setSize*/) {

            /*if (lAvg / setSize > 255 || (aAvg / setSize) > 255 || bAvg / setSize > 255) {
                cout << "Averages are fucked";
            }*/
            int setSize = associatedPixels.size();
            averageColor = cv::Vec3f(lAvg / (float)setSize,  aAvg / (float)setSize, bAvg / (float)setSize);
            position = cv::Point2f(x / setSize, y / setSize);
        }

        //Come back here tomorrow
        void addPixel(Pixel pix) {
            associatedPixels.push_back(pix);
            //Add color here too it just makes sense
            x += pix.position.x;
            y += pix.position.y;

            lAvg += pix.color[0];
            aAvg += pix.color[1];
            bAvg += pix.color[2];
        }
        /*m * sqrt(N / (float)M)*/

        double differenceCost(Pixel pix, double diffCoeff) {
            return ColorDifference(pix.color, paletteColor) + diffCoeff * PositionDifference(pix.position, position);
        }

        void clearAssociated() {
            associatedPixels.clear();
            lAvg = 0;
            aAvg = 0;
            bAvg = 0;

            x = 0;
            y = 0;
        }

        void normalizeProbs(double normFactor) {
            for (int i = 0; i < probToCluster.size(); i++) {
                probToCluster[i] = probToCluster[i] / normFactor;
            }
        }


    private:

        double lAvg = 0;
        double aAvg = 0;
        double bAvg = 0;

        double x = 0;
        double y = 0;


};

class PaletteColor {
    public:
        double weight;
        cv::Vec3f color;
        vector<Pixel> associatedPixels;


        void setWeight(double p) {
            weight = p;
        }

        void copy(PaletteColor copyColor) {

            color = copyColor.color;
        //C compiler apparently automatically copies it lol
            associatedPixels = copyColor.associatedPixels;
            weight = copyColor.weight;
        }

        void calculateCenter() {
            cv::Vec3d avg = cv::Vec3d(0, 0, 0);

            for (int i = 0; i < associatedPixels.size(); i++) {
                avg += associatedPixels[i].color;
            }

            avg = avg / (double)associatedPixels.size();

            color = avg;
        }

        void perturb(cv::Vec3d direction, double alpha) {
            color[0] += alpha * direction[0];
            color[1] += alpha * direction[1];
            color[2] += alpha * direction[1];
        }

        void updateWeight(vector<SuperPixel>* sPixels, int clusterIndex, double probSuperPixel) {
            double tempWeight = 0.0;
            for (int s = 0; s < (*sPixels).size(); s++) {
                tempWeight += (*sPixels)[s].probToCluster[clusterIndex] * probSuperPixel;
            }
            weight = tempWeight;
        }

        cv::PCA PCA() {
            //Data as row, or {[l1, a1, b1], [l2, a2, b2], [l3, a3, b3]}
            cv::Mat data(associatedPixels.size(), 3, CV_64F);
            //cv::Mat mean(associatedPixels.size(), 3, CV_32F);
            for (int i = 0; i < data.rows; i++) {
                data.at<double>(i, 0) = associatedPixels[i].color[0] / 100.0;
                data.at<double>(i, 1) = associatedPixels[i].color[1] / 127;
                data.at<double>(i, 2) = associatedPixels[i].color[2] / 127;
            }

            return cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
        }

        cv::Vec3f average(PaletteColor color2) {
            return (color + color2.color) / 2;
        }

};





cv::Mat bilateralFilterMat(vector<SuperPixel>* sPixels, int wOut, int hOut) {
    cv::Mat averages(hOut, wOut, CV_32FC3); 
    cv::Mat bilatFilter(hOut, wOut, CV_32FC3);
    for (int r = 0; r < hOut; r++) {
        for (int c = 0; c < wOut; c++) {
            *averages.ptr<cv::Vec3f>(r, c) = (*sPixels)[c + r * wOut].averageColor;
        }
    }

    cv::bilateralFilter(averages, bilatFilter, 5, 50, 50);
    return bilatFilter;

}


vector<cv::Point2f> LaplachianSmooth(vector<SuperPixel>* sPixels, float movePercentage, int wOut, int hOut) {

    cv::Point2i regions[] = { cv::Point2i(0,1), cv::Point2i(1,0), cv::Point2i(0,-1), cv::Point2i(-1, 0) };
    vector<cv::Point2f> output;
    for (int r = 0; r < hOut; r++) {
        for (int c = 0; c < wOut; c++) {
            SuperPixel& focusPixel = (*sPixels)[c + r * wOut];

            cv::Point2f avg(0, 0);
            int dividend = 0;
            for (int i = 0; i < 4; i++) {
                //Its so yucky that Mat is (r,c) bc it makes this randomly need to be the opposite even tho x,y MAKES MORE SENSE
                //HATE U MATLAB RAH
                cv::Point2i checkPixel = cv::Point2i(c, r) + regions[i];
                if (inBounds(checkPixel, wOut, hOut)) {
                    /*cv::Point2f& neighbor = (*sPixels)[checkPixel.x + checkPixel.y * wOut].position;
                    avg.x += neighbor.x;
                    avg.y += neighbor.y;*/

                    avg = avg + (*sPixels)[checkPixel.x + checkPixel.y * wOut].position;

                    dividend++;
                }
            }
            avg = avg / dividend;



            cv::Point2f newPos = focusPixel.position + movePercentage * (/*new*/avg - /*old*/focusPixel.position);
            output.push_back(newPos);
        }
    }

    return output;



}


double probabiltyBelongsToCluster(SuperPixel* sPixel, PaletteColor* c, cv::Vec3f bilatColor, double temperature) {
    double p = (*c).weight * exp(-1 * ColorDifference(bilatColor, (*c).color) / temperature);
    (*sPixel).probToCluster.push_back(p);
    return p;
}


PaletteColor initialize(const cv::Mat& sourceMat, vector<SuperPixel>* sPixels, vector<PaletteColor>* palette, int wOut, int hOut) {
    
    

    /*cv::imshow("", sourceMat);
    cv::waitKey(0);*/
    PaletteColor firstColor;
    //Size of the superPixel in the origin image
    int fullSuperPixelWidth = (sourceMat).cols / wOut;
    int fullSuperPixelHeight = (sourceMat).rows / hOut;


    //Initializes the superPixel centers
    int rowCounter = 0;
    int colCounter = 0;

    int sPixelCount = wOut * hOut;
    for (int i = 0; i < sPixelCount; i++) {
        SuperPixel sPixel;
        sPixel.position.x = (fullSuperPixelWidth * colCounter) + (fullSuperPixelWidth / 2);
        sPixel.position.y = (fullSuperPixelHeight * rowCounter) + (fullSuperPixelHeight / 2);

        colCounter++;

        if (colCounter >= wOut) {
            colCounter = 0;
            rowCounter++;
        }
        sPixels->push_back(sPixel);
    }


    //Associates regular pixels to a superpixel, while also calculating their average
    for (int r = 0; r < sourceMat.rows; r++) {
        for (int c = 0; c < sourceMat.cols; c++) {
            
            //Columns/rows relative position as a percentage multiplied by the output size
            //When truncated gives the fullSize pixel's place in the shrunken image
            int cOut = (int)((c / (float)sourceMat.cols) * wOut);
            int rOut = (int)((r / (float)sourceMat.rows) * hOut);  
            ////from 1D array to 2D space
            int currentIndex = cOut + rOut * wOut;
            
            cv::Point2i testPoint(c, r);
            
            cv::Vec3f pix = *sourceMat.ptr<cv::Vec3f>(r, c);

            Pixel pixelObj;
            pixelObj.color = pix;
            pixelObj.position = testPoint;

            (*sPixels)[currentIndex].associatedPixels.push_back(pixelObj);
            (*sPixels)[currentIndex].addToAvg(pix[0], pix[1], pix[2]);

            firstColor.associatedPixels.push_back(pixelObj);
 
        }
    }

    //Set the Average the color of every superPixel (bad coding) 
    for (int i = 0; i < wOut * hOut; i++) {

        (*sPixels)[i].setAvg();
        (*sPixels)[i].paletteColor = firstColor.color;
    }

    return firstColor;
    //firstColor.setWeight(0.5f);

}


 void AssosciatePixToSuperPix(const cv::Mat& sourceMat, vector<SuperPixel>* sPixels, double diffCoeff, int wOut, int hOut) {
    //Assign pixels to SuperPixel

    //double* distanceToSP = (double*)malloc(wOut * hOut * sizeof(double));
    int N = wOut * hOut;
    int M = sourceMat.rows * sourceMat.cols;
    //Clear the data  
    for (int i = 0; i < N; i++) {
        /*(*sPixels)[i].associatedPixels.clear();*/
        (*sPixels)[i].clearAssociated();
    }

    cv::Point2i regions[] = { cv::Point2i(0,1), cv::Point2i(1,1), cv::Point2i(1,0), cv::Point2i(1,-1), 
        cv::Point2i(0,-1), cv::Point2i(-1,-1), cv::Point2i(-1, 0), cv::Point2i(-1,1)};


    //Associate Pixels and calculate mean
    int counter = 0;
    for (int r = 0; r < sourceMat.rows; r++) {
        for (int c = 0; c < sourceMat.cols; c++) {

            Pixel currentPix;
            currentPix.position.x = c;
            currentPix.position.y = r;
            currentPix.color = *(sourceMat.ptr<cv::Vec3f>(r, c));

            int cOut = (int)((c / (float)sourceMat.cols) * wOut);
            int rOut = (int)((r / (float)sourceMat.rows) * hOut);

            int index = cOut + rOut * wOut;
            int minIndex = index;
            double minDistance = (*sPixels)[index].differenceCost(currentPix, diffCoeff);

            for (int i = 1; i < 8; i++) {
                if (inBounds(cOut + regions[i].x, rOut + regions[i].y, wOut, hOut)) {
                    index = cOut + regions[i].x + (rOut + regions[i].y) * wOut;
                    double dist = (*sPixels)[index].differenceCost(currentPix, diffCoeff);
                    if (dist < minDistance) {
                        minDistance = dist;
                        minIndex = index;
                    }
                }
            }

            (*sPixels)[minIndex].addPixel(currentPix);

        }
    }

    //Set the average 
    for (int i = 0; i < wOut * hOut; i++) {
        (*sPixels)[i].setAvg();
    }
    

    //free(distanceToSP);
}

void CalculateProbabilities(const cv::Mat& bilatFilter, vector<SuperPixel>* sPixels, vector<PaletteColor>* palette, double temperature, int wOut, int hOut) {
    //Calculate P(Ck | Ps)
    for (int r = 0; r < hOut; r++) {
        for (int c = 0; c < wOut; c++) {
            SuperPixel& sPixel = (*sPixels)[c + r * wOut];
            sPixel.probToCluster.clear();

            cv::Vec3f bilat = *bilatFilter.ptr<cv::Vec3f>(r, c);

            double max = probabiltyBelongsToCluster(&sPixel, &(*palette)[0], bilat, temperature);
            int maxIndx = 0;
            double sumNorm = max;
            int paletteSize = (*palette).size();
            for (int i = 1; i < paletteSize; i++) {
                double temporary = probabiltyBelongsToCluster(&sPixel, &(*palette)[i], bilat, temperature);
                sumNorm += temporary;
                if (temporary > max) {
                    max = temporary;
                    maxIndx = i;
                }
            }

            sPixel.normalizeProbs(sumNorm);


            sPixel.paletteColor = (*palette)[maxIndx].color;
        }
    }

    //Recalculate P(Ck)
    double sizeD = (double)(*sPixels).size();
    for (int i = 0; i < (*palette).size(); i++) {
        (*palette)[i].updateWeight(sPixels, i, 1 / sizeD);
    }
}

void PaletteRefine(const cv::Mat& bilatFilter, vector<SuperPixel>* sPixels, vector<PaletteColor>* palette, vector<cv::Vec3f>* oldPalette, int wOut, int hOut) {

    oldPalette->clear();

    double probSuperPixel = 1 / (double)(wOut * hOut);
    
    //Where i is the cluster index
    //c + r*wOut is the sPixel index
    for (int i = 0; i < (*palette).size(); i++) {

        cv::Vec3f refinedColor(0, 0, 0);

        for (int r = 0; r < hOut; r++){ 
            for (int c = 0; c < wOut; c++) {

                cv::Vec3f m = bilatFilter.at<cv::Vec3f>(r, c);
                double probSuperPixelToCluster = (*sPixels)[c + r * wOut].probToCluster[i];
                refinedColor += m * probSuperPixelToCluster * probSuperPixel;

            }
        }
        (*oldPalette).push_back((*palette)[i].color);
        (*palette)[i].color = refinedColor / (*palette)[i].weight;
    }

}
 
double PaletteChange(vector<cv::Vec3f>* oldPalette, vector<PaletteColor>* newPalette) {
    double sumDiff = 0;
    for (int i = 0; i < (*oldPalette).size(); i++) {
        sumDiff += ColorDifference((*oldPalette)[i], (*newPalette)[i].color);
        /*cout << sumDiff;
        cout << "+";*/
    }
    /*cout << "=";
    cout << sumDiff;
    cout << "\n";*/

    return sumDiff;
}


void MergeSubClusters(vector<PaletteColor>* palette, vector<Cluster>* clusters) {
    vector<PaletteColor> finalPalette;
    for (int c = 0; c < clusters->size(); c++) {
        Cluster& cluster = (*clusters)[c];
        PaletteColor finalColor;
        finalColor.copy((*palette)[cluster.sub1Index]);
        finalColor.weight += (*palette)[cluster.sub2Index].weight;
        finalPalette.push_back(finalColor);
    }

    *palette = finalPalette;
}

void SplitCluster(vector<PaletteColor>* palette, vector<Cluster>* clusters, double epsilon, int clusterIndex) {

    
    
}

void PaletteExpand(vector<PaletteColor>* palette, vector<Cluster>* clusters, double epsilon, int maxPalette) {
    //cached up here so that the changing size doesn't affect for loop
    int size = clusters->size();

    //Check for splits
    for (int c = 0; c < size; c++) {

        if ((*clusters).size() == maxPalette) {
            MergeSubClusters(palette, clusters);
            break;
        }

        Cluster& cluster = (*clusters)[c];

        PaletteColor& ck1 = (*palette)[cluster.sub1Index];
        PaletteColor& ck2 = (*palette)[cluster.sub2Index];

        double split = ColorDifference((*palette)[cluster.sub1Index].color, (*palette)[cluster.sub2Index].color);
        //cout << split;
        //cout << "\n";
        if (split > epsilon) {

            //cout << ck1.associatedPixels.size();
            //Create a new cluster to store Ck2
            //Ck1 stays as first cluster object 
            Cluster newCluster;
            newCluster.sub1Index = cluster.sub2Index;


            //Save copy of associated pixels, split between newly formed clusters
            vector<Pixel> fullCluster = ck1.associatedPixels;
            ck1.associatedPixels.clear();
            ck2.associatedPixels.clear();

            int clusterSize = fullCluster.size();
            for (int i = 0; i < clusterSize; i++) {
                Pixel p = fullCluster[i];
                double ck1Diff = ColorDifference(p.color, ck1.color);
                double ck2Diff = ColorDifference(p.color, ck2.color);

                if (ck1Diff > ck2Diff) {
                    ck2.associatedPixels.push_back(p);
                }
                else {
                    ck1.associatedPixels.push_back(p);
                }
            }

            //Copy of ck1
            PaletteColor ck1Copy;
            ck1.setWeight(ck1.weight / 2);
            ck1Copy.copy(ck1);
            (*palette).push_back(ck1Copy);
            //Set ck2 index to last index since it was just pushed to back
            cluster.sub2Index = (*palette).size() - 1;
            cluster.color = ck1.color;

            PaletteColor ck2Copy;
            //ck2Copy.copy(ck2);

            PaletteColor& ck2 = (*palette)[newCluster.sub1Index];
            ck2.setWeight(ck2.weight / 2);
            ck2Copy.copy(ck2);
            (*palette).push_back(ck2Copy);
            newCluster.sub2Index = (*palette).size() - 1;
            newCluster.color = ck2.color;

            (*clusters).push_back(newCluster);
        }

        //SplitCluster(palette, clusters, epsilon, c);
    }

    

    //Move subclusters so they can split in subsequent convergences
    int clustersSize = clusters->size();
    for (int c = 0; c < clustersSize; c++) {
        Cluster& cluster = (*clusters)[c];
        PaletteColor& ck1 = (*palette)[cluster.sub1Index];
        PaletteColor& ck2 = (*palette)[cluster.sub2Index];


        cv::PCA pcAnalysis = ck1.PCA();
        cv::Vec3f principleAxis = pcAnalysis.eigenvectors.at<cv::Vec3f>(0);
        //ck1.perturb(principleAxis, 2);
        ck2.perturb(principleAxis, 2);
        
    }

}

void ShowSuperPixelOutput() {

}



int main(int argc, char* args[])
{

    bool DEBUG = true;

    int wOut = 23;
    int hOut = 32;

    int paletteSize = 8;

    cv::Mat inputImage = cv::imread("Obamna.png", cv::IMREAD_COLOR);

    //cv::imshow("1", inputImage);
    //cv::waitKey();

    cv::Mat floatImage(inputImage.rows, inputImage.cols, CV_32FC3);
    inputImage.convertTo(floatImage, CV_32F, 1.0/255);

    //cv::imshow("2", floatImage);
    //cv::waitKey();

    cv::Mat croppedImage = floatImage(cv::Range(0, (floatImage.cols / wOut) * wOut), cv::Range(0, (floatImage.rows / hOut) * hOut));
    
    //cv::imshow("3", croppedImage);
    //cv::waitKey();

    cv::Mat sourceMat(croppedImage.rows, croppedImage.cols, CV_32FC3);
    cv::cvtColor(croppedImage, sourceMat, cv::COLOR_BGR2Lab);
    
    int N = wOut * hOut;
    int M = sourceMat.rows * sourceMat.cols;
    double diffCoeff = 45 * sqrt(N / (double)M);
   /* cv::imshow("4", sourceMat);
    cv::waitKey();*/


    //Deconstruct to free data

    
    

    //cv::Rect crop(0, 0, (floatImage.cols / wOut) * wOut, (floatImage.rows / hOut) * hOut);
    //cv::Rect crop(cv::Range(0, (floatImage.cols / wOut) * wOut), cv::Range(0, (floatImage.rows / hOut) * hOut));
    //floatImage = floatImage(cv::Range(0, (floatImage.cols / wOut) * wOut), cv::Range(0, (floatImage.rows / hOut) * hOut));

    

    //floatImage.~Mat();

   //cv::Mat sourceMat(floatImage.rows, floatImage.cols, CV_32F);

    


    vector<Cluster> clusters;
    vector<PaletteColor> palette;
    vector<SuperPixel> superPixels;


    


    //Initialize Superclusters and initial cluster
    PaletteColor criticalCluster =  initialize(sourceMat, &superPixels, &palette, wOut, hOut);

    

    //Initialize cluster weights/sub cluster
    criticalCluster.setWeight(0.5f);
    criticalCluster.calculateCenter();

    cv::PCA wholeImagePCA = criticalCluster.PCA();
    
    PaletteColor protoCopy;
    protoCopy.copy(criticalCluster);
    
    cv::Vec3f principleAxis = cv::Vec3f(wholeImagePCA.eigenvectors.at<float>(0,0), wholeImagePCA.eigenvectors.at<float>(0, 1), wholeImagePCA.eigenvectors.at<float>(0, 2));
    
    principleAxis *= 1.5;


    /*if (DEBUG) {
        cout << principleAxis[0];
        cout << "\n";
        cout << principleAxis[1];
        cout << "\n";
        cout << principleAxis[2];
    }*/

    criticalCluster.perturb(principleAxis, 1);
    protoCopy.perturb(principleAxis, -1);

    palette.push_back(criticalCluster);
    palette.push_back(protoCopy);

    Cluster c1;
    c1.color = criticalCluster.average(protoCopy);
    c1.sub1Index = 0;
    c1.sub2Index = 1;
    clusters.push_back(c1);

    float max = wholeImagePCA.eigenvalues.at<double>(0);

    double temperature = 1.1 * (max);
    double finalTemp = 1.0;
    double alpha = 0.7;
    
    //If total palette change is less than var, palette has converged
    float paletteEpsilon = 1.0f;
    float clusterEpsilon = 0.25f; 

    

    //MAIN LOOP

    cv::Mat bilatFilter(hOut, wOut, CV_32FC3);
    vector<cv::Point2f> newPoints;
    vector<cv::Vec3f> oldColors;

    int loopCounter = 0;
    while (temperature > finalTemp) {
       /* if (loopCounter == ) {
            cout << "Breaks here";
        }*/

        if (DEBUG) {
            cout << temperature;
            cout << "\n";
        }
        //SuperPixels
        //If statement is unneccesary since 0 < 0 is false 
        if (newPoints.size() != 0) {
            for (int i = 0; i < superPixels.size(); i++) {
                superPixels[i].position = newPoints[i];
            }
        }
        AssosciatePixToSuperPix(sourceMat, &superPixels, diffCoeff, wOut, hOut);
        newPoints = LaplachianSmooth(&superPixels, 0.4f, wOut, hOut);
        bilatFilter = bilateralFilterMat(&superPixels, wOut, hOut);

        


        cv::imwrite("bilateral.png", bilatFilter);

        //SuperPixel/Palette Association
        CalculateProbabilities(bilatFilter, &superPixels, &palette, temperature, wOut, hOut);

        //Refine and Expand
        PaletteRefine(bilatFilter, &superPixels, &palette, &oldColors, wOut, hOut);

        

        //If palette convereged
        double change = PaletteChange(&oldColors, &palette);
        
        if (change < paletteEpsilon) {
            
            //Lower Temperature
            temperature = alpha * temperature;

            if (clusters.size() < paletteSize) {
                PaletteExpand(&palette, &clusters, clusterEpsilon, paletteSize);
            }

            loopCounter++;

        }
        /*cout << loopCounter;
        cout << "\n";*/
        
    }

    double beta = 1.1;
    cv::Mat labOutput(hOut, wOut, CV_32FC3);

    for (int r = 0; r < hOut; r++) {
        for (int c = 0; c < wOut; c++) {
            SuperPixel& sPixel = superPixels[c + r * wOut];
            labOutput.at<cv::Vec3f>(r, c) = cv::Vec3f(sPixel.paletteColor[0], sPixel.paletteColor[1] * beta, sPixel.paletteColor[2] * beta);
        }
    }

    cv::Mat output(hOut, wOut, CV_32FC3);
    cv::cvtColor(labOutput, output, cv::COLOR_Lab2BGR);

    cv::imwrite("Output.png", output);

    /*cv::imshow("Pixelated Image", output);
    cv::waitKey();*/





    

    
    
    //Split First palette color into 2
    



   

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
