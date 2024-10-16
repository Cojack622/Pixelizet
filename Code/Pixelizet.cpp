// Pixels.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/interface.h>





//Has the potential to make a REALLY GOOD obama
#define PixelAssociateSearchDiameter 7
#define colorDistanceWeight 45
#define alpha 0.7
#define paletteEpsTest 2.0f
//Controls how far apart two clusters have to be before they are considered to be different
#define clusterEpsTest 0.5f
#define finalTemp 1.0
#define bilatFilterDiameter 3
#define bilatFilterAlpha 15
#define palSize 8
//Creates a outputScale x y image where y aligns with the input image's aspect ratio
#define outScale 64 
#define inFile "input/itsAlive.png"
#define numThreads 7


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

struct HyperParameters {
    int searchDiameter;
    int bilateralFilterDiameter;
    int bilateralFilterAlpha;
    float clusterEpsilon;
    float paletteEpsilon;
    

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

        /*static double diffCoeff;*/

        cv::Vec3f averageColor;
        cv::Vec3f paletteColor;
        std::vector<Pixel> associatedPixels;
        std::vector<double> probToCluster;
        
        cv::Point2f position;

        void addToAvg(double l, double a, double b) {
            lAvg += l;
            aAvg += a;
            bAvg += b;
        }

        void setAvg(/*int setSize*/) {


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
        std::vector<Pixel> associatedPixels;


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

        void perturb(cv::Vec3d direction, double mod) {
            color[0] += mod * direction[0];
            color[1] += mod * direction[1];
            color[2] += mod * direction[2];
        }

        void updateWeight(std::vector<SuperPixel>* sPixels, int clusterIndex, double probSuperPixel) {
            double tempWeight = 0.0;
            for (int s = 0; s < (*sPixels).size(); s++) {
                tempWeight += (*sPixels)[s].probToCluster[clusterIndex] * probSuperPixel;
            }
            weight = tempWeight;
        }

        cv::PCA PCA() {
            //Data as row, or {[l1, a1, b1], [l2, a2, b2], [l3, a3, b3]}

            //Later manually calculate Mean vector to save time
            cv::Mat data(associatedPixels.size(), 3, CV_32F);
            
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





double diffCoeff;
int hOut, wOut;


cv::Mat bilateralFilterMat(std::vector<SuperPixel>* sPixels) {
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


std::vector<cv::Point2f> LaplachianSmooth(std::vector<SuperPixel>* sPixels, float movePercentage) {

    cv::Point2i regions[] = { cv::Point2i(0,1), cv::Point2i(1,0), cv::Point2i(0,-1), cv::Point2i(-1, 0) };
    std::vector<cv::Point2f> output;
    for (int r = 0; r < hOut; r++) {
        for (int c = 0; c < wOut; c++) {
            SuperPixel& focusPixel = (*sPixels)[c + r * wOut];

            cv::Point2f avg(0, 0);
            int dividend = 0;
            for (int i = 0; i < 4; i++) {
                //Its so yucky that Mat is (r,c) bc it makes this randomly need to be the opposite even tho x,y MAKES MORE SENSE
                //HATE U OPENCV RAH
                cv::Point2i checkPixel = cv::Point2i(c, r) + regions[i];
                if (inBounds(checkPixel, wOut, hOut)) {

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


PaletteColor initialize(const cv::Mat& sourceMat, std::vector<SuperPixel>* sPixels, std::vector<PaletteColor>* palette) {
    
    


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

    //Set the Average the color of every superPixel (more efficient to do it inside previous per-pixel loop) 
    for (int i = 0; i < wOut * hOut; i++) {

        (*sPixels)[i].setAvg();
        (*sPixels)[i].paletteColor = firstColor.color;
    }

    return firstColor;

}


void AssociateRegionToSuperPix(const cv::Mat& sourceMat, std::vector<SuperPixel>* sPixels, std::vector<cv::Point2i>* regions, std::mutex* writeMtx, int rStart, int cStart) {
    
    //This is dumb
    int regionR = sourceMat.rows / numThreads;
    int regionC = sourceMat.cols / numThreads;

    int rEnd = rStart + regionR;
    int cEnd = cStart + regionC;
    if (rEnd < sourceMat.rows && sourceMat.rows < rStart + 2 * regionR) {
        rEnd = sourceMat.rows;
    }
    if (cEnd < sourceMat.cols && sourceMat.cols < cStart + 2 * regionC) {
        cEnd = sourceMat.cols;
    }


    for (int r = rStart; r < rEnd; r++) {
        for (int c = cStart; c < cEnd; c++) {

            Pixel currentPix;
            currentPix.position.x = c;
            currentPix.position.y = r;
            currentPix.color = *(sourceMat.ptr<cv::Vec3f>(r, c));

            int cOut = (int)((c / (float)sourceMat.cols) * wOut);
            int rOut = (int)((r / (float)sourceMat.rows) * hOut);

            int index = cOut + rOut * wOut;
            int minIndex = index;
            double minDistance = (*sPixels)[index].differenceCost(currentPix, diffCoeff);

            for (int i = 1; i < (*regions).size(); i++) {
                if (inBounds(cOut + (*regions)[i].x, rOut + (*regions)[i].y, wOut, hOut)) {
                    index = cOut + (*regions)[i].x + (rOut + (*regions)[i].y) * wOut;
                    double dist = (*sPixels)[index].differenceCost(currentPix, diffCoeff);
                    if (dist < minDistance) {
                        minDistance = dist;
                        minIndex = index;
                    }
                }
            }

            (*writeMtx).lock();
            (*sPixels)[minIndex].addPixel(currentPix);
            (*writeMtx).unlock();
        }
    }
}


 void AssosciatePixToSuperPix(const cv::Mat& sourceMat, std::vector<SuperPixel>* sPixels, std::vector<cv::Point2i>* regions) {
    //Assign pixels to SuperPixel

    //double* distanceToSP = (double*)malloc(wOut * hOut * sizeof(double));
    int N = wOut * hOut;
    int M = sourceMat.rows * sourceMat.cols;
    //Clear the data  
    for (int i = 0; i < N; i++) {
        /*(*sPixels)[i].associatedPixels.clear();*/
        (*sPixels)[i].clearAssociated();
    }


    int rSize = sourceMat.rows / numThreads;
    int cSize = sourceMat.cols / numThreads;

    std::thread threads[numThreads];
    std::mutex writeMtx;

    for (int r = 0; r < numThreads; r++) {
        for (int c = 0; c < numThreads; c++) {
            threads[c] = std::thread(AssociateRegionToSuperPix, sourceMat, sPixels, regions, &writeMtx, r * rSize, c * cSize);
        }
        
        for (int c = 0; c < numThreads; c++) {
            threads[c].join();
        }
    }


    //Set the average 
    for (int i = 0; i < N; i++) {
        (*sPixels)[i].setAvg();
    }
    
}

void CalculateProbabilities(const cv::Mat& bilatFilter, std::vector<SuperPixel>* sPixels, std::vector<PaletteColor>* palette, double temperature) {
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

void PaletteRefine(const cv::Mat& bilatFilter, std::vector<SuperPixel>* sPixels, std::vector<PaletteColor>* palette, std::vector<cv::Vec3f>* oldPalette) {

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
 
double PaletteChange(std::vector<cv::Vec3f>* oldPalette, std::vector<PaletteColor>* newPalette) {
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


void MergeSubClusters(std::vector<PaletteColor>* palette, std::vector<Cluster>* clusters) {
    std::vector<PaletteColor> finalPalette;
    for (int c = 0; c < clusters->size(); c++) {
        Cluster& cluster = (*clusters)[c];
        PaletteColor finalColor;
        finalColor.copy((*palette)[cluster.sub1Index]);
        finalColor.weight += (*palette)[cluster.sub2Index].weight;
        finalPalette.push_back(finalColor);
    }

    *palette = finalPalette;
}

void PaletteExpand(std::vector<PaletteColor>* palette, std::vector<Cluster>* clusters, double epsilon, int maxPalette) {
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
            std::vector<Pixel> fullCluster = ck1.associatedPixels;
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


        if(ck1.associatedPixels.size() > 0){
            cv::PCA pcAnalysis = ck1.PCA();
            cv::Vec3f principleAxis = cv::Vec3f(pcAnalysis.eigenvectors.at<float>(0, 0), pcAnalysis.eigenvectors.at<float>(0, 1), pcAnalysis.eigenvectors.at<float>(0, 2));

            ck1.perturb(principleAxis, -1);
            ck2.perturb(principleAxis, 1);
        }

        
    }

}

void ShowSuperPixelOutput(std::vector<SuperPixel>* sPixels, int convergance, int wOut, int hOut) {
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


int GetInputParams(int argc, char* args[], std::string* inputFile, int* outputScale, int* paletteSize) {
    if (argc != 4 ) {
        printf("Please format command as \"Pixelizet file_path outputScale paletteSize");
        //printf("Or Pixelizet outputScale paletteSize ")
        printf("Please make sure paletteSize is no more than 16");
        return -1;
    }

    std::string imageName = args[1];
    *inputFile = "input/" + imageName;
        

    *outputScale = std::stoi(args[2]);
    *paletteSize = std::stoi(args[3]);

    if (*paletteSize > 16 || *paletteSize < 0) {
        printf("Please make sure paletteSize is no more than 16");
        return -1;
    }

    return 0;
}

cv::Mat GetInputMat(std::string inputFile, int outputScale) {
    cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_COLOR);

    if (inputImage.cols > inputImage.rows) {
        wOut = outputScale;
        hOut = outputScale * (inputImage.rows / (float)inputImage.cols);
    }
    else if (inputImage.rows > inputImage.rows) {
        hOut = outputScale;
        wOut = outputScale * (inputImage.cols / (float)inputImage.rows);
    }
    else {
        hOut = outputScale;
        wOut = outputScale;
    }

    //Get File Params

    cv::Mat floatImage(inputImage.rows, inputImage.cols, CV_32FC3);
    inputImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    inputImage.release();


    cv::Mat sourceMat(floatImage.rows, floatImage.cols, CV_32FC3);
    cv::cvtColor(floatImage, sourceMat, cv::COLOR_BGR2Lab);
    floatImage.release();

    return sourceMat;
}


//Returns initial temp
double InitializeSuperPixelsAndClusters(cv::Mat& sourceMat, std::vector<Cluster>* clusters, std::vector<PaletteColor>* palette, std::vector<SuperPixel>* superPixels) {
    //Initialize SuperPixels and initial cluster
    PaletteColor criticalCluster =  initialize(sourceMat, superPixels, palette);    

    //Initialize cluster weights/sub cluster
    criticalCluster.setWeight(0.5f);
    criticalCluster.calculateCenter();

    cv::PCA wholeImagePCA = criticalCluster.PCA();
    
    PaletteColor protoCopy;
    protoCopy.copy(criticalCluster);
    
    cv::Vec3f principleAxis = cv::Vec3f(wholeImagePCA.eigenvectors.at<float>(0,0), wholeImagePCA.eigenvectors.at<float>(0, 1), wholeImagePCA.eigenvectors.at<float>(0, 2));

    criticalCluster.perturb(principleAxis, 1);
    protoCopy.perturb(principleAxis, -1);

    palette->push_back(criticalCluster);
    palette->push_back(protoCopy);

    Cluster c1;
    c1.color = criticalCluster.average(protoCopy);
    c1.sub1Index = 0;
    c1.sub2Index = 1;
    clusters->push_back(c1);

    //getting Eigenvalue as a double breaks the entire program since it makes it within (0, 1) range   
    return wholeImagePCA.eigenvalues.at<float>(0) / 19.0;
}

std::vector<cv::Point2i> GenerateSearchVector() {
    int searchBoxDiameter = PixelAssociateSearchDiameter;
    std::vector<cv::Point2i> regions;
    for (int i = -(searchBoxDiameter / 2); i < (searchBoxDiameter / 2) + 1; i++) {
        for (int j = -(searchBoxDiameter / 2); j < (searchBoxDiameter / 2) + 1; j++) {

            if (i != 0 || j != 0) {
                regions.push_back(cv::Point2i(i, j));
            }

        }
    }

    return regions;
}

//void GetHyperParameters(std::string paramsFile, ) {}



int main(int argc, char* args[])
{

    bool DEBUG = true;
    
    std::string inputFile;
    std::string outputFile = "output/output.png";
    int outputScale, paletteSize;
    if (!DEBUG) {
        int gotParams = GetInputParams(argc, args, &inputFile, &outputScale, &paletteSize);
        if (gotParams < 0) return -1;
    }
    else {
        inputFile = inFile;
        outputScale = outScale;
        paletteSize = palSize;
    }


    cv::Mat sourceMat = GetInputMat(inputFile, outputScale);

    int N = wOut * hOut;
    int M = sourceMat.rows * sourceMat.cols;
    diffCoeff = colorDistanceWeight * sqrt(N / (double)M);


    std::vector<Cluster> clusters;
    std::vector<PaletteColor> palette;
    std::vector<SuperPixel> superPixels;

    double temperature = InitializeSuperPixelsAndClusters(sourceMat, &clusters, &palette, &superPixels);
    //Save copy of MaxTemp to use as completion marker
    double maxTemp = temperature;

    std::vector<cv::Point2i> regions = GenerateSearchVector();


    

    cv::Mat bilatFilter(hOut, wOut, CV_32FC3);
    std::vector<cv::Point2f> newPoints;
    std::vector<cv::Vec3f> oldColors;

    //MAIN LOOP
    while (temperature > finalTemp) {
        //SuperPixels
        //If statement is unneccesary since 0 < 0 is false 
        if (newPoints.size() != 0) {
            for (int i = 0; i < superPixels.size(); i++) {
                superPixels[i].position = newPoints[i];
            }
        }
        AssosciatePixToSuperPix(sourceMat, &superPixels, &regions);
        newPoints = LaplachianSmooth(&superPixels, 0.4f);
        bilatFilter = bilateralFilterMat(&superPixels);

        //SuperPixel/Palette Association
        CalculateProbabilities(bilatFilter, &superPixels, &palette, temperature);

        //Refine and Expand
        PaletteRefine(bilatFilter, &superPixels, &palette, &oldColors);

        //If palette convereged
        double change = PaletteChange(&oldColors, &palette);
        
        if (change < paletteEpsTest) {   

            //Temperature is not a very good metric for completion since convergence towards the end makes much longer
            std::cout << 100 * (1 - temperature / maxTemp);
            std::cout << "%\n";

            //Lower Temperature
            temperature = alpha * temperature;

            if (clusters.size() < paletteSize) {
                PaletteExpand(&palette, &clusters, clusterEpsTest, paletteSize);
            }

        }

    }



    sourceMat.release();
    bilatFilter.release();

    float beta = 1.1;
    cv::Mat labOutput(hOut, wOut, CV_8UC3);

    for (int r = 0; r < hOut; r++) {
        for (int c = 0; c < wOut; c++) {
            SuperPixel& sPixel = superPixels[c + r * wOut];
            labOutput.at<cv::Vec3b>(r, c) = cv::Vec3b((unsigned char)(sPixel.paletteColor[0] * (255.0/100.0)), (unsigned char) (sPixel.paletteColor[1] * beta - 128), (unsigned char) (sPixel.paletteColor[2] * beta - 128));

        }
    }

    cv::Mat rgbOutput(hOut, wOut, CV_8UC3);
    cv::cvtColor(labOutput, rgbOutput, cv::COLOR_Lab2BGR);

    cv::Mat rgbLarge(hOut * 10, wOut * 10, CV_8UC3);
    
    for (int r = 0; r < rgbLarge.rows; r++) {
        for (int c = 0; c < rgbLarge.cols; c++) {
            int cOut = (int)((c / (float)rgbLarge.cols) * wOut);
            int rOut = (int)((r / (float)rgbLarge.rows) * hOut);
            rgbLarge.at<cv::Vec3b>(r, c) = rgbOutput.at<cv::Vec3b>(rOut, cOut);
        }
    }

    cv::imwrite("output/outputLarge.png", rgbLarge);
    cv::imwrite(outputFile, rgbOutput);

    return 0;
}