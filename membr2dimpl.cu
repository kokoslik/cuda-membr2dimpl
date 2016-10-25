#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#define EPS 1e-6
#define WRITE_TO_FILE

#define PHI_CPU(x,y) sin(M_PI*x)*sin(M_PI*y)
#define PSI_CPU(x,y) 0
#define F_CPU(x,y,t) 0
using namespace std;

//Обработчик ошибок
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( error ) (HandleError( error, __FILE__, __LINE__ ))

//float solveGPU(double L, double T, double tau, int N)
//{
//#ifdef WRITE_TO_FILE
//    ofstream ofile("../datagpu.dat");
//    ofile.precision(16);
//    int counter=0, writeeach=1;
//#endif
//    cudaEvent_t start,stop;
//    float gputime=0;
//    HANDLE_ERROR( cudaEventCreate(&start) );
//    HANDLE_ERROR( cudaEventCreate(&stop) );
//    double *U,*Unext,*Uloc,*F,*err;
//    double *M,*Mdev,*Fdev,*errdev;
//    int Nn=N+1;
//    int Nplus=Nn+2;
//    double h=L/N,t=0.0;
//    size_t size=Nplus*sizeof(double);
//    size_t sizeM=3*Nn*sizeof(double);
//    F=new double[Nplus];
//    Uloc=new double[Nplus];
//    err=new double[Nplus];
//    M=new double[Nn*3];
//    double maxerr;
//    HANDLE_ERROR( cudaMalloc(&U,size) );
//    HANDLE_ERROR( cudaMalloc(&Unext,size) );
//    HANDLE_ERROR( cudaMalloc(&Mdev,sizeM) );
//    HANDLE_ERROR( cudaMalloc(&Fdev,size) );
//    HANDLE_ERROR( cudaMalloc(&errdev,size) );
//    thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(errdev);
//    M[0]=0.0;
//    M[1]=1.0;
//    M[2]=0.0;
//    for(int i=1;i<Nn-1;i++)
//    {
//        M[i*3]=-tau/(h*h);
//        M[i*3+1]=1.0+2.0*tau/(h*h);
//        M[i*3+2]=-tau/(h*h);
//    }
//    M[(Nn-1)*3]=-2.0*tau/(h*h);
//    M[(Nn-1)*3+1]=1.0+2.0*tau/(h*h);
//    M[(Nn-1)*3+2]=0.0;
//    HANDLE_ERROR( cudaMemcpy(Mdev,M,sizeM,cudaMemcpyHostToDevice) );
//    HANDLE_ERROR( cudaMemset(U,0,size) );
//    memset(Uloc,0,size);
//    dim3 threads(1024,1,1),blocks(Nn%1024==0?Nn/1024:Nn/1024+1,1,1);
//    HANDLE_ERROR( cudaEventRecord(start) );
//    while(t<T-0.5*tau)
//    {
//        HANDLE_ERROR( cudaMemcpy(Fdev,U,size,cudaMemcpyDeviceToDevice) );
//        double a=0.0;
//        double b=5.0*2.0*tau/h+Uloc[N+1];
//        HANDLE_ERROR( cudaMemcpy(&Fdev[1],&a,sizeof(double),cudaMemcpyHostToDevice) );
//        HANDLE_ERROR( cudaMemcpy(&Fdev[N+1],&b,sizeof(double),cudaMemcpyHostToDevice) );
//        do{
//                iter_kernel<<<blocks,threads>>>(U,Unext,Mdev,Fdev,Nn,errdev);
//                HANDLE_ERROR( cudaGetLastError() );
//                HANDLE_ERROR( cudaDeviceSynchronize() );
//                thrust::device_ptr<double> max_ptr = thrust::max_element(err_ptr+1, err_ptr + Nn+1);
//                maxerr=max_ptr[0];
//                swap(U,Unext);
//        }while(maxerr>EPS);

//        t+=tau;
//#ifdef WRITE_TO_FILE
//        if(counter%writeeach==0)
//        {
//            HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
//            for(int i=0;i<Nn;i++)
//            ofile<<Uloc[i+1]<<endl;
//            ofile<<endl;
//            ofile<<endl;
//        }
//        counter++;
//#endif
//    }
//    HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
//    HANDLE_ERROR( cudaEventRecord(stop) );
//    HANDLE_ERROR( cudaEventSynchronize(stop) );
//    HANDLE_ERROR( cudaEventElapsedTime(&gputime,start,stop) );
//#ifdef WRITE_TO_FILE
//    ofile.close();
//#endif

//    delete[] Uloc;
//    delete[] err;
//    delete[] M;
//    delete[] F;
//    HANDLE_ERROR( cudaFree(U) );
//    HANDLE_ERROR( cudaFree(Unext) );
//    HANDLE_ERROR( cudaFree(Mdev) );
//    HANDLE_ERROR( cudaFree(Fdev) );
//    HANDLE_ERROR( cudaFree(errdev) );
//    HANDLE_ERROR( cudaEventDestroy(start) );
//    HANDLE_ERROR( cudaEventDestroy(stop) );
//    return 1e-3*gputime;
//}

float solveCPU(double a,double L1,double L2,double T,double tau,int N1,int N2)
{
#ifdef WRITE_TO_FILE
    ofstream ofile("../datacpu.dat");
    ofile.precision(16);
#endif
    float cputime=0;
    double *Uprev,*U,*Unext,*F;
    double *M;
    int N1n=N1+1,N2n=N2+1;
    double h1=L1/N1,h2=L2/N2,t=tau;
    Uprev=new double[N1n*N2n];
    U=new double[N1n*N2n];
    F=new double[N1n*N2n];
    Unext=new double[N1n*N2n];
    M=new double[N1n*N2n*5];
    double maxerr;
    //Нулевой временной слой
        for(int i=0;i<N1n;i++)
        {
            for(int j=0;j<N2n;j++)
            {
                Uprev[i*N2n+j]=PHI_CPU(i*h1,j*h2);
    #ifdef WRITE_TO_FILE
                ofile<<Uprev[i*N2n+j]<<' ';
    #endif
            }
#ifdef WRITE_TO_FILE
            ofile<<endl;
#endif
        }
#ifdef WRITE_TO_FILE
    ofile<<endl;
    ofile<<endl;
#endif
    cputime=clock();
    //Первый временной слой
//    for(int i=0;i<N1n;i++)
//        for(int j=0;j<N2n;j++)
//            if((i==0)||(j==0)||(i==N1)||(j==N2))
//                {
//                    M[(i*N2n+j)*5]=0.0;
//                    M[(i*N2n+j)*5+1]=0.0;
//                    M[(i*N2n+j)*5+2]=1.0;
//                    M[(i*N2n+j)*5+3]=0.0;
//                    M[(i*N2n+j)*5+4]=0.0;
//                    F[i*N2n+j]=Uprev[i*N2n+j];
//                }
//            else
//                {
//                    M[(i*N2n+j)*5]=-0.5*tau*tau*a*a/(h1*h1);
//                    M[(i*N2n+j)*5+1]=-0.5*tau*tau*a*a/(h2*h2);
//                    M[(i*N2n+j)*5+2]=tau*tau*a*a*(1.0/(h1*h1)+1.0/(h2*h2))+1.0;
//                    M[(i*N2n+j)*5+3]=-0.5*tau*tau*a*a/(h2*h2);
//                    M[(i*N2n+j)*5+4]=-0.5*tau*tau*a*a/(h1*h1);
//                    F[i*N2n+j]=-Uprev[i*N2n+j]-tau*PSI_CPU(i*h1,j*h2)-tau*tau*0.5*F_CPU(i*h1,j*h2,tau)+a*a*tau*tau*((PHI_CPU((i+1)*h1,j*h2)-2.0*PHI_CPU(i*h1,j*h2)+PHI_CPU((i-1)*h1,j*h2))/(h1*h1)+(PHI_CPU(i*h1,(j+1)*h2)-2.0*PHI_CPU(i*h1,j*h2)+PHI_CPU(i*h1,(j-1)*h2))/(h2*h2));
//                }
//    memcpy(U,Uprev,N1n*N2n*sizeof(double));
//    do{
//        maxerr=0;
//        for(int i=1;i<N1n-1;i++)
//            for(int j=1;j<N2n-1;j++)
//            Unext[i*N2n+j]=U[i*N2n+j]+1.0/M[(i*N2n+j)*5+2]*(F[i*N2n+j]-M[(i*N2n+j)*5]*U[(i-1)*N2n+j]-M[(i*N2n+j)*5+1]*U[i*N2n+j-1]-M[(i*N2n+j)*5+2]*U[i*N2n+j]-M[(i*N2n+j)*5+3]*U[i*N2n+j+1]-M[(i*N2n+j)*5+4]*U[(i+1)*N2n+j]);
//        for(int i=0;i<N1n*N2n;i++)
//        {
//            double err=abs(Unext[i]-U[i]);
//            if(err>maxerr)maxerr=err;
//        }
//        swap(U,Unext);
//    }while(maxerr>EPS);

        for(int i=0;i<N1n;i++)
        {
            for(int j=0;j<N2n;j++)
            {
                if((i==0)||(j==0)||(i==N1)||(j==N2))
                {
                    U[i*N2n+j]=Uprev[i*N2n+j];
                    Unext[i*N2n+j]=Uprev[i*N2n+j];
                }
                else
                {
                    U[i*N2n+j]=Uprev[i*N2n+j]+tau*PSI_CPU(i*h1,j*h2)+
                            tau*tau*0.5*F_CPU(i*h1,j*h2,0.0)+
                            a*a*tau*tau*0.5*((PHI_CPU((i+1)*h1,j*h2)-2.0*PHI_CPU(i*h1,j*h2)+PHI_CPU((i-1)*h1,j*h2))/(h1*h1)+(PHI_CPU(i*h1,(j+1)*h2)-2.0*PHI_CPU(i*h1,j*h2)+PHI_CPU(i*h1,(j-1)*h2))/(h2*h2));
                }}}

#ifdef WRITE_TO_FILE
        for(int i=0;i<N1n;i++)
        {
            for(int j=0;j<N2n;j++)
                ofile<<U[i*N2n+j]<<' ';
            ofile<<endl;
        }
        ofile<<endl;
        ofile<<endl;
#endif
    //Следующие временные слои
    for(int i=0;i<N1n;i++)
        for(int j=0;j<N2n;j++)
            if((i==0)||(j==0)||(i==N1)||(j==N2))
                {
                    M[(i*N2n+j)*5]=0.0;
                    M[(i*N2n+j)*5+1]=0.0;
                    M[(i*N2n+j)*5+2]=1.0;
                    M[(i*N2n+j)*5+3]=0.0;
                    M[(i*N2n+j)*5+4]=0.0;
                }
            else
                {
                    M[(i*N2n+j)*5]=a*a/(h1*h1);
                    M[(i*N2n+j)*5+1]=a*a/(h2*h2);
                    M[(i*N2n+j)*5+2]=-2.0*a*a*(1.0/(h1*h1)+1.0/(h2*h2))-1.0/(tau*tau);
                    M[(i*N2n+j)*5+3]=a*a/(h2*h2);
                    M[(i*N2n+j)*5+4]=a*a/(h1*h1);
                }
    while(t<T-0.5*tau)
    {
        for(int i=1;i<N1n-1;i++)
            for(int j=1;j<N2n-1;j++)
                        F[i*N2n+j]=Uprev[i*N2n+j]/(tau*tau)-2.0*U[i*N2n+j]/(tau*tau)-F_CPU(i*h1,j*h2,t+tau);
        memcpy(Uprev,U,N1n*N2n*sizeof(double));
        do{
            maxerr=0;
            for(int i=1;i<N1n-1;i++)
                for(int j=1;j<N2n-1;j++)
                Unext[i*N2n+j]=Uprev[i*N2n+j]+1.0/M[(i*N2n+j)*5+2]*(F[i*N2n+j]-M[(i*N2n+j)*5]*Uprev[(i-1)*N2n+j]-M[(i*N2n+j)*5+1]*Uprev[i*N2n+j-1]-M[(i*N2n+j)*5+2]*Uprev[i*N2n+j]-M[(i*N2n+j)*5+3]*Uprev[i*N2n+j+1]-M[(i*N2n+j)*5+4]*Uprev[(i+1)*N2n+j]);
            for(int i=0;i<N1n*N2n;i++)
            {
                double err=abs(Unext[i]-Uprev[i]);
                if(err>maxerr)maxerr=err;
            }
            swap(Uprev,Unext);
        }while(maxerr>EPS);
        t+=tau;
        swap(Uprev,U);
#ifdef WRITE_TO_FILE
        for(int i=0;i<N1n;i++)
        {
            for(int j=0;j<N2n;j++)
                ofile<<U[i*N2n+j]<<' ';
            ofile<<endl;
        }
        ofile<<endl;
        ofile<<endl;
#endif
    }
    cputime=(double)(clock()-cputime)/CLOCKS_PER_SEC;
#ifdef WRITE_TO_FILE
    ofile.close();
#endif

    delete[] Uprev;
    delete[] U;
    delete[] Unext;
    delete[] M;
    delete[] F;
    return cputime;
}

int main()
{
    float gpu,gpushared,cpu;
    //gpu=solveGPU(1.0,50.0,0.01,1000000);
    //cout<<"GPU Time: "<<gpu<<endl;
    //gpushared=solveGPUshared(1.0,50.0,0.01,1000000);
    //cout<<"GPU Time: "<<gpushared<<endl;
    cpu=solveCPU(1.0,1.0,1.0,100,0.05,25,25);
    cout<<"CPU Time: "<<cpu<<endl;
    //cout<<"Max ratio:"<<cpu/min(gpu,gpushared)<<endl;
    return 0;
}
