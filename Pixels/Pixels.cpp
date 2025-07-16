// Pixels.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/interface.h>

using namespace std; 

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

        void setAvg() {
            int setSize = associatedPixels.size();
            averageColor = cv::Vec3f(lAvg / (float)setSize,  aAvg / (float)setSize, bAvg / (float)setSize);
            position = cv::Point2f(x / setSize, y / setSize);
        }

        void addPixel(Pixel pix) {
            associatedPixels.push_back(pix);
            
            x += pix.position.x;
            y += pix.position.y;

            lAvg += pix.color[0];
            aAvg += pix.color[1];
            bAvg += pix.color[2];
        }

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
            //Compiler should copy by value here
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

        void perturb(cv::Vec3d direction, double mod) {
            color[0] += mod * direction[0];
            color[1] += mod * direction[1];
            color[2] += mod * direction[2];
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
            cv::Mat data(associatedPixels.size(), 3, CV_32F);
            
            //cv::Mat mean(associatedPixels.size(), 3, CV_32F);
            for (int i = 0; i < data.rows; i++) {
                data.at<float>(i, 0) = associatedPixels[i].color[0];
                data.at<float>(i, 1) = associatedPixels[i].color[1];
                data.at<float>(i, 2) = associatedPixels[i].color[2];
            }

            return cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
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

    cv::bilateralFilter(averages, bilatFilter, bilatFilterDiameter, bilatFilterAlpha, bilatFilterAlpha);
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
    //In future do some dynamic programming here to avoid doing a full pass through SuperPixels
    for (int i = 0; i < wOut * hOut; i++) {

        (*sPixels)[i].setAvg();
        (*sPixels)[i].paletteColor = firstColor.color;
    }

    return firstColor;
}


 void AssosciatePixToSuperPix(const cv::Mat& sourceMat, vector<SuperPixel>* sPixels, double diffCoeff, int wOut, int hOut) {
    //Assign pixels to SuperPixel

    int N = wOut * hOut;
    int M = sourceMat.rows * sourceMat.cols;

    //Clear the data  
    for (int i = 0; i < N; i++) {
        (*sPixels)[i].clearAssociated();
    }

    int searchBoxDiameter = 5;
    vector<cv::Point2i> regions;
    for (int i = -(searchBoxDiameter / 2); i < (searchBoxDiameter / 2) + 1; i++) {
        for (int j = -(searchBoxDiameter / 2); j < (searchBoxDiameter / 2) + 1; j++) {
            
            if (i != 0 || j != 0) {
                regions.push_back(cv::Point2i(i, j));
            }
        }
    }

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

            for (int i = 1; i < regions.size(); i++) {
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
            int pSize = (*palette).size();
            for (int i = 1; i < pSize; i++) {
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

        if (split > epsilon) {

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


        if(ck1.associatedPixels.size() > 0){
            cv::PCA pcAnalysis = ck1.PCA();
            cv::Vec3f principleAxis = cv::Vec3f(pcAnalysis.eigenvectors.at<float>(0, 0), pcAnalysis.eigenvectors.at<float>(0, 1), pcAnalysis.eigenvectors.at<float>(0, 2));

            ck1.perturb(principleAxis, -1);
            ck2.perturb(principleAxis, 1);
        }

        
    }

}

void ShowSuperPixelOutput(vector<SuperPixel>* sPixels, int convergance, int wOut, int hOut) {
    cv::Mat spOutput = cv::Mat::zeros(hOut, wOut, CV_32FC3);

    for (int i = 0; i < (*sPixels).size(); i++) {
        SuperPixel& superPixel = (*sPixels)[i];
        cv::Vec3f checkColor(0.0f, 0.0f, 0.0f);
        cv::Vec3f randColor(rand() % 10, rand() % 10, rand() % 10);
        for (int j = 0; j < superPixel.associatedPixels.size(); j++) {
            Pixel& p = superPixel.associatedPixels[j];
            

            cv::Vec3f color = spOutput.at<cv::Vec3f>(p.position);
            
            if (color == checkColor) {
                spOutput.at<cv::Vec3f>(p.position) = superPixel.paletteColor + randColor;
            }
            else {
                spOutput.at<cv::Vec3f>(p.position) = cv::Vec3f(74.93, 23.94, 78.96);
            }


        }
    }

    
    cv::Mat showMat(spOutput.rows, spOutput.cols, CV_32FC3);
    cv::cvtColor(spOutput, showMat, cv::COLOR_Lab2BGR);
    cv::imshow("Palette Image", showMat);
    cv::waitKey();
}



int main(int argc, char* args[])
{

    bool DEBUG = true;
    
    string outputFile = "test_output/output.png";
    
    cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_COLOR);

    float scale = outputScale / 1080.0;
    int wOut = (int)(inputImage.cols * scale);
    int hOut = (int)(inputImage.rows * scale);

    cv::Mat floatImage(inputImage.rows, inputImage.cols, CV_32FC3);
    inputImage.convertTo(floatImage, CV_32F, 1.0/255);

    cv::Mat croppedImage = floatImage(cv::Range(0, (floatImage.rows / hOut) * hOut), cv::Range(0, (floatImage.cols / wOut) * wOut));

    cv::Mat sourceMat(croppedImage.rows, croppedImage.cols, CV_32FC3);
    cv::cvtColor(croppedImage, sourceMat, cv::COLOR_BGR2Lab);
    
    int N = wOut * hOut;
    int M = sourceMat.rows * sourceMat.cols;
    double diffCoeff = colorDistanceWeight * sqrt(N / (double)M);


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

    criticalCluster.perturb(principleAxis, 1);
    protoCopy.perturb(principleAxis, -1);

    palette.push_back(criticalCluster);
    palette.push_back(protoCopy);

    Cluster c1;
    c1.color = criticalCluster.average(protoCopy);
    c1.sub1Index = 0;
    c1.sub2Index = 1;
    clusters.push_back(c1);

    float max = 1.1 * wholeImagePCA.eigenvalues.at<float>(0);
    double temperature = max;

    //MAIN LOOP

    cv::Mat bilatFilter(hOut, wOut, CV_32FC3);
    vector<cv::Point2f> newPoints;
    vector<cv::Vec3f> oldColors;

    int loopCounter = 0;
    while (temperature > finalTemp) {
        //SuperPixels
        //If statement is unneccesary since 0 < 0 is false 
        //But just in case
        if (newPoints.size() != 0) {
            for (int i = 0; i < superPixels.size(); i++) {
                superPixels[i].position = newPoints[i];
            }
        }
        AssosciatePixToSuperPix(sourceMat, &superPixels, diffCoeff, wOut, hOut);
        newPoints = LaplachianSmooth(&superPixels, 0.4f, wOut, hOut);
        bilatFilter = bilateralFilterMat(&superPixels, wOut, hOut);

        //SuperPixel/Palette Association
        CalculateProbabilities(bilatFilter, &superPixels, &palette, temperature, wOut, hOut);

        //Refine and Expand
        PaletteRefine(bilatFilter, &superPixels, &palette, &oldColors, wOut, hOut);

        //If palette convereged
        double change = PaletteChange(&oldColors, &palette);
        
        if (change < paletteEpsilon) {   

            cout << 100 * (1 - temperature / max);
            cout << "\n";
            
            //Lower Temperature
            temperature = alpha * temperature;

            if (clusters.size() < paletteSize) {
                //ShowSuperPixelOutput(&superPixels, loopCounter, sourceMat.cols, sourceMat.rows);
                PaletteExpand(&palette, &clusters, clusterEpsilon, paletteSize);
            }
            else {
                cout << "Got all clusters";
            }

            loopCounter++;

        }
        
    }

    double beta = 1.1;
    cv::Mat labOutput(hOut, wOut, CV_8UC3);

    for (int r = 0; r < hOut; r++) {
        for (int c = 0; c < wOut; c++) {
            SuperPixel& sPixel = superPixels[c + r * wOut];
            labOutput.at<cv::Vec3b>(r, c) = cv::Vec3b((unsigned char)(sPixel.paletteColor[0] * (255.0/100.0)), (unsigned char) (sPixel.paletteColor[1] * beta - 128), (unsigned char) (sPixel.paletteColor[2] * beta - 128));

        }
    }

    cv::Mat paletteOutput(1, palette.size(), CV_8UC3);
    for (int p = 0; p < palette.size(); p++) {
        paletteOutput.at<cv::Vec3b>(0, p) = cv::Vec3b((unsigned char)(palette[p].color[0] * (255.0 / 100.0)), (unsigned char)(palette[p].color[1] * beta - 128), (unsigned char)(palette[p].color[2] * beta - 128));
    }

    cv::Mat rgbOutput(hOut, wOut, CV_8UC3);
    cv::cvtColor(labOutput, rgbOutput, cv::COLOR_Lab2BGR);

    cv::Mat rgbPalette(1, palette.size(), CV_8UC3);
    cv::cvtColor(paletteOutput, rgbPalette, cv::COLOR_Lab2BGR);

    cv::imwrite("Palette.png", rgbPalette);
    cv::imwrite(outputFile, rgbOutput);

    return 0;
}
