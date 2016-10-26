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
#define EPS 1e-3
//#define WRITE_TO_FILE
#define NX 32
#define NY 32
#define PHI_CPU(x,y) sin(M_PI*x)*sin(M_PI*y)
#define PHI_GPU(x,y) __sinf(M_PI*x)*__sinf(M_PI*y)
#define PSI_CPU(x,y) 0
#define PSI_GPU(x,y) 0
#define F_CPU(x,y,t) 0
#define F_GPU(x,y,t) 0
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



__global__ void first_layer_kernel(double *U,double *Uprev,double tau,double a, int N1n,int N2n, double h1, double h2)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x+1;
    int j=threadIdx.y+blockIdx.y*blockDim.y+1;
    if((i < N1n-1)&&(j<N2n-1))
        U[i*N2n+j]=Uprev[i*N2n+j]+tau*PSI_GPU(i*h1,j*h2)+
                tau*tau*0.5*F_GPU(i*h1,j*h2,0.0)+
                a*a*tau*tau*0.5*((PHI_GPU((i+1)*h1,j*h2)-2.0*PHI_GPU(i*h1,j*h2)+PHI_GPU((i-1)*h1,j*h2))/(h1*h1)+(PHI_GPU(i*h1,(j+1)*h2)-2.0*PHI_GPU(i*h1,j*h2)+PHI_GPU(i*h1,(j-1)*h2))/(h2*h2));

}

__global__ void calc_F_kernel(double *U,double *Uprev,double *F,double tau,double t, int N1n,int N2n, double h1, double h2)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x+1;
    int j=threadIdx.y+blockIdx.y*blockDim.y+1;
    if((i < N1n-1)&&(j<N2n-1))
        F[i*N2n+j]=Uprev[i*N2n+j]/(tau*tau)-2.0*U[i*N2n+j]/(tau*tau)-F_GPU(i*h1,j*h2,t+tau);

}

__global__ void iter_kernel(double *Unext,double *Uprev,double *F,double *M,double *errdev, int N1n,int N2n)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x+1;
    int j=threadIdx.y+blockIdx.y*blockDim.y+1;
    if((i < N1n-1)&&(j<N2n-1))
    {
        Unext[i*N2n+j]=Uprev[i*N2n+j]+1.0/M[(i*N2n+j)*5+2]*(F[i*N2n+j]-M[(i*N2n+j)*5]*Uprev[(i-1)*N2n+j]-M[(i*N2n+j)*5+1]*Uprev[i*N2n+j-1]-M[(i*N2n+j)*5+2]*Uprev[i*N2n+j]-M[(i*N2n+j)*5+3]*Uprev[i*N2n+j+1]-M[(i*N2n+j)*5+4]*Uprev[(i+1)*N2n+j]);
        errdev[i*N2n+j]=abs(Unext[i*N2n+j]-Uprev[i*N2n+j]);
    }

}

double solveGPU(double a,double L1,double L2,double T,double tau,int N1,int N2)
{
    double *Unext,*U,*Uprev,*Uloc,*errdev,*M,*Mdev,*F;
    double h1=L1/N1,h2=L2/N2,maxerr=0;
    int N1n=N1+1,N2n=N2+1;
    double t=tau;
    float gputime=0.0;
    size_t size=N1n*N2n*sizeof(double);
    dim3 threads(NX,NY,1),blocks((N1-1)%NX==0?(N1-1)/NX:(N1-1)/NX+1,(N2-1)%NY==0?(N2-1)/NY:(N2-1)/NY+1,1);
    Uloc=new double[N1n*N2n];
    M=new double[N1n*N2n*5];
    HANDLE_ERROR( cudaMalloc(&U,size) );
    HANDLE_ERROR( cudaMalloc(&Unext,size) );
    HANDLE_ERROR( cudaMalloc(&Uprev,size) );
    HANDLE_ERROR( cudaMalloc(&Mdev,size*5) );
    HANDLE_ERROR( cudaMalloc(&errdev,size) );
    HANDLE_ERROR( cudaMalloc(&F,size) );
    thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(errdev);
#ifdef WRITE_TO_FILE
    ofstream ofile("../datagpu.dat");
    ofile.precision(16);
#endif
    //Нулевой временной слой
    for(int i=0;i<N1n;i++)
    {
        for(int j=0;j<N2n;j++)
        {
            Uloc[i*N2n+j]=PHI_CPU(i*h1,j*h2);
#ifdef WRITE_TO_FILE
            ofile<<Uloc[i*N2n+j]<<' ';
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
    HANDLE_ERROR( cudaMemcpy(Uprev,Uloc,size,cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(U,Uprev,size,cudaMemcpyDeviceToDevice) );
    HANDLE_ERROR( cudaMemcpy(Unext,Uprev,size,cudaMemcpyDeviceToDevice) );
    //Первый временной слой
    cudaEvent_t start,stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start) );
    first_layer_kernel<<<blocks,threads>>>(U,Uprev,tau,a,N1n,N2n,h1,h2/*,phi,psi,f*/);
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
#ifdef WRITE_TO_FILE
    HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
    for(int i=0;i<N1n;i++)
    {
        for(int j=0;j<N2n;j++)
            ofile<<Uloc[i*N2n+j]<<' ';
        ofile<<endl;
    }
    ofile<<endl;
    ofile<<endl;
#endif

    //Формирование матрицы системы
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
HANDLE_ERROR( cudaMemcpy(Mdev,M,5*size,cudaMemcpyHostToDevice) );
    //Основной цикл
    while(t<T-0.5*tau)
    {
        calc_F_kernel<<<blocks,threads>>>(U,Uprev,F,tau,t,N1n,N2n,h1,h2);
        HANDLE_ERROR( cudaGetLastError() );
        HANDLE_ERROR( cudaDeviceSynchronize() );
        HANDLE_ERROR( cudaMemcpy(Uprev,U,size,cudaMemcpyDeviceToDevice) );
        do{

              iter_kernel<<<blocks,threads>>>(Unext,Uprev,F,Mdev,errdev,N1n,N2n);
              HANDLE_ERROR( cudaGetLastError() );
              HANDLE_ERROR( cudaDeviceSynchronize() );
              thrust::device_ptr<double> max_ptr = thrust::max_element(err_ptr+1, err_ptr + N1n*N2n-1);
              maxerr=max_ptr[0];
              swap(Uprev,Unext);
                }while(maxerr>EPS);
#ifdef WRITE_TO_FILE
        HANDLE_ERROR( cudaMemcpy(Uloc,Unext,size,cudaMemcpyDeviceToHost) );
        for(int i=0;i<N1n;i++)
        {
            for(int j=0;j<N2n;j++)
                ofile<<Uloc[i*N2n+j]<<' ';
            ofile<<endl;
        }
        ofile<<endl;
        ofile<<endl;
#endif

        t+=tau;
        swap(Uprev,U);
    }
    HANDLE_ERROR( cudaMemcpy(Uloc,Unext,size,cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaEventRecord(stop) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&gputime,start,stop) );
    HANDLE_ERROR( cudaFree(U) );
    HANDLE_ERROR( cudaFree(Unext) );
    HANDLE_ERROR( cudaFree(Uprev) );
    HANDLE_ERROR( cudaFree(Mdev) );
    HANDLE_ERROR( cudaFree(errdev) );
    HANDLE_ERROR( cudaFree(F) );
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );
    delete[] Uloc;
    delete[] M;
#ifdef WRITE_TO_FILE
    ofile.close();
#endif
    return (double)gputime/1000.0;
}

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
    float gpu,cpu;
    cpu=solveCPU(1.0,1.0,1.0,10,0.01,500,500);
    cout<<"CPU Time: "<<cpu<<endl;
    gpu=solveGPU(1.0,1.0,1.0,10,0.01,500,500);
    cout<<"GPU Time: "<<gpu<<endl;
    cout<<"Max ratio:"<<cpu/gpu<<endl;
    return 0;
}
