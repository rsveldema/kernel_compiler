#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

inline void check_hip(hipError_t status, const char* label) {
    if (status != hipSuccess) { std::fprintf(stderr, "%s: %s\n", label, hipGetErrorString(status)); std::abort(); }
}
inline void check_hipblas(hipblasStatus_t status, const char* label) {
    if (status != HIPBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: hipBLAS status %d\n", label, static_cast<int>(status)); std::abort(); }
}
struct HipDimension { uint32_t x, y, z; };
struct VStatistics { std::string kernel_name; size_t host_to_device=0, device_to_host=0, host_to_device_bytes=0, device_to_host_bytes=0; };
enum class KernelDimension { OneD, TwoD, ThreeD };
enum class KernelType { Vector, Matrix, Triangular };
class AbstractKernel;

class ComputeKernelRegistry {
public:
    class ScopedActiveKernel {
    public: explicit ScopedActiveKernel(AbstractKernel&); ~ScopedActiveKernel();
    private: AbstractKernel* previous_=nullptr;
    };
    static ComputeKernelRegistry& instance() { static ComputeKernelRegistry r; return r; }
    void registerKernel(AbstractKernel&); void recordHostToDevice(AbstractKernel*,size_t); void recordDeviceToHost(AbstractKernel*,size_t);
    void recordHostToDevice(std::string_view,size_t); void recordDeviceToHost(std::string_view,size_t); void resetStatistics();
    void enableRegistrationLog(const std::string&); std::vector<VStatistics> getStatistics(int);
    static AbstractKernel* activeKernel() { return slot(); } static std::string_view activeKernelName();
private:
    static AbstractKernel*& slot() { thread_local AbstractKernel* k=nullptr; return k; }
    void log(const AbstractKernel&); std::vector<AbstractKernel*> kernels_; std::vector<std::string> logged_; std::string log_file_;
};

class AbstractKernel {
public:
    AbstractKernel(std::string n, KernelDimension d=KernelDimension::OneD, KernelType t=KernelType::Vector, std::string desc={})
      : name_(std::move(n)), dimension_(d), type_(t), descriptor_(std::move(desc)) { ComputeKernelRegistry::instance().registerKernel(*this); }
    virtual ~AbstractKernel()=default; const std::string& kernelName() const{return name_;} const std::string& generatedDescriptor()const{return descriptor_;}
    void recordHostToDevice(size_t n){++stats_.host_to_device;stats_.host_to_device_bytes+=n;} void recordDeviceToHost(size_t n){++stats_.device_to_host;stats_.device_to_host_bytes+=n;}
    void resetStatistics(){stats_={};} VStatistics statistics()const{auto s=stats_;s.kernel_name=name_;return s;}
    static std::string demangleKernelName(const std::string& n){auto p=n.find('|');return p==std::string::npos?n:n.substr(p+1);}
private: std::string name_; KernelDimension dimension_; KernelType type_; std::string descriptor_; VStatistics stats_;
};

inline ComputeKernelRegistry::ScopedActiveKernel::ScopedActiveKernel(AbstractKernel& k):previous_(slot()){slot()=&k;}
inline ComputeKernelRegistry::ScopedActiveKernel::~ScopedActiveKernel(){slot()=previous_;}
inline void ComputeKernelRegistry::registerKernel(AbstractKernel& k){if(std::find(kernels_.begin(),kernels_.end(),&k)==kernels_.end()){kernels_.push_back(&k);log(k);}}
inline void ComputeKernelRegistry::recordHostToDevice(AbstractKernel*k,size_t n){if(k)k->recordHostToDevice(n);} inline void ComputeKernelRegistry::recordDeviceToHost(AbstractKernel*k,size_t n){if(k)k->recordDeviceToHost(n);}
inline void ComputeKernelRegistry::recordHostToDevice(std::string_view s,size_t n){for(auto*k:kernels_)if(k->kernelName()==s){k->recordHostToDevice(n);break;}}
inline void ComputeKernelRegistry::recordDeviceToHost(std::string_view s,size_t n){for(auto*k:kernels_)if(k->kernelName()==s){k->recordDeviceToHost(n);break;}}
inline void ComputeKernelRegistry::resetStatistics(){for(auto*k:kernels_)k->resetStatistics();}
inline std::string_view ComputeKernelRegistry::activeKernelName(){return activeKernel()?std::string_view(activeKernel()->kernelName()):std::string_view{};}
inline std::vector<VStatistics> ComputeKernelRegistry::getStatistics(int n){std::vector<VStatistics> r;for(auto*k:kernels_)r.push_back(k->statistics());std::sort(r.begin(),r.end(),[](auto&a,auto&b){return a.host_to_device_bytes+a.device_to_host_bytes>b.host_to_device_bytes+b.device_to_host_bytes;});if(n>=0&&size_t(n)<r.size())r.resize(n);return r;}
inline void ComputeKernelRegistry::enableRegistrationLog(const std::string& f){log_file_=f;logged_.clear();std::ofstream(f,std::ios::trunc)<<"compute kernels\nname\ttiled\n";for(auto*k:kernels_)log(*k);}
inline void ComputeKernelRegistry::log(const AbstractKernel& k){if(log_file_.empty()||std::find(logged_.begin(),logged_.end(),k.kernelName())!=logged_.end())return;std::ofstream(log_file_,std::ios::app)<<AbstractKernel::demangleKernelName(k.kernelName())<<'\t'<<k.generatedDescriptor()<<'\n';logged_.push_back(k.kernelName());}

class HipQueue {
public:
    explicit HipQueue(size_t=0){check_hip(hipStreamCreateWithFlags(&stream_,hipStreamNonBlocking),"hipStreamCreateWithFlags");check_hipblas(hipblasCreate(&blas_),"hipblasCreate");check_hipblas(hipblasSetStream(blas_,stream_),"hipblasSetStream");}
    ~HipQueue(){if(blas_)(void)hipblasDestroy(blas_);if(stream_)(void)hipStreamDestroy(stream_);} HipQueue(const HipQueue&)=delete; HipQueue&operator=(const HipQueue&)=delete;
    hipStream_t stream()const{return stream_;} std::recursive_mutex&mutex()const{return mutex_;}
    hipblasHandle_t blas_handle()const{return blas_;}
    void wait(const char*label="HIP stream synchronization"){check_hip(hipStreamSynchronize(stream_),label);}
private: hipStream_t stream_=nullptr; hipblasHandle_t blas_=nullptr; mutable std::recursive_mutex mutex_;
};

class HipSession {
public:
    explicit HipSession(bool=false,const char* preferred_device_name=nullptr); HipSession(const HipSession&)=delete; HipSession&operator=(const HipSession&)=delete;
    bool has_device()const{return has_device_;} int get_device()const{return device_;} size_t queue_count()const{return queues_.size();} size_t num_queues()const{return queue_count();}
    HipQueue&get_queue(size_t i){assert(!queues_.empty());return*queues_[i%queues_.size()];} size_t maxMemoryAllocationSize()const{return max_allocation_;}
    bool shader_buffer_float32_atomic_add_enabled()const{return true;} bool cooperative_matrix2_enabled()const{return false;}
    const std::string&cooperative_matrix2_unavailable_reason()const{return coop_reason_;}
private: int device_=0;bool has_device_=false;size_t max_allocation_=0;std::string coop_reason_="HIP cooperative-matrix lowering is not enabled";std::vector<std::unique_ptr<HipQueue>>queues_;
};
class HipComputeContext { public: explicit HipComputeContext(HipSession&s):session_(s){} operator HipQueue&()const{return session_.get_queue(0);} private:HipSession&session_; };

class VBaseHostBuffer {
public:
    VBaseHostBuffer(HipSession&s,size_t n):session_(s),size_(n){check_hip(hipHostMalloc(&data_,n),"hipHostMalloc");} virtual~VBaseHostBuffer(){if(data_)(void)hipHostFree(data_);}
    VBaseHostBuffer(const VBaseHostBuffer&)=delete;VBaseHostBuffer&operator=(const VBaseHostBuffer&)=delete;
    VBaseHostBuffer(VBaseHostBuffer&&o)noexcept:session_(o.session_),data_(std::exchange(o.data_,nullptr)),size_(std::exchange(o.size_,0)){}
    VBaseHostBuffer&operator=(VBaseHostBuffer&&o)noexcept{assert(&session_==&o.session_);if(this!=&o){if(data_)(void)hipHostFree(data_);data_=std::exchange(o.data_,nullptr);size_=std::exchange(o.size_,0);}return*this;}
    virtual const char*typed_buffer_name()const=0;uint8_t*get(){return static_cast<uint8_t*>(data_);}const uint8_t*get()const{return static_cast<const uint8_t*>(data_);}size_t size()const{return size_;}HipSession&session()const{return session_;}
private:HipSession&session_;void*data_=nullptr;size_t size_=0;
};
template<typename T>class VHostBuffer:public VBaseHostBuffer {
public:using value_type=std::conditional_t<std::is_array_v<T>,std::remove_extent_t<T>,T>;explicit VHostBuffer(HipSession&s):VBaseHostBuffer(s,sizeof(T)){}const char*typed_buffer_name()const override{return"VHostBuffer";}
    value_type*get(){return reinterpret_cast<value_type*>(VBaseHostBuffer::get());}const value_type*get()const{return reinterpret_cast<const value_type*>(VBaseHostBuffer::get());}value_type get(size_t i)const{return get()[i];}void set(size_t i,value_type v){get()[i]=v;}
    template<typename U>void fill(U v){std::fill_n(get(),count(),static_cast<value_type>(v));}template<typename U>void fill(U v,size_t n){std::fill_n(get(),n,static_cast<value_type>(v));}template<typename U>void fill_at(U v,size_t o,size_t n){std::fill_n(get()+o,n,static_cast<value_type>(v));}size_t count()const{return std::is_array_v<T>?std::extent_v<T>:1;}
};
class VDynamicHostBuffer:public VBaseHostBuffer{public:VDynamicHostBuffer(HipSession&s,size_t n):VBaseHostBuffer(s,n){}const char*typed_buffer_name()const override{return"VDynamicHostBuffer";}uint8_t*bytes(){return get();}const uint8_t*bytes()const{return get();}};

class VBaseDeviceBuffer {
public:
    VBaseDeviceBuffer(HipSession&,size_t n):size_(n){check_hip(hipMalloc(&data_,n),"hipMalloc");}virtual~VBaseDeviceBuffer(){if(data_)(void)hipFree(data_);}VBaseDeviceBuffer(const VBaseDeviceBuffer&)=delete;VBaseDeviceBuffer&operator=(const VBaseDeviceBuffer&)=delete;virtual const char*typed_buffer_name()const=0;
    void*get()const{return data_;}size_t size()const{return size_;}
    void write(HipQueue&q,VBaseHostBuffer&s,size_t n=SIZE_MAX,size_t so=0,size_t d=0){if(n==SIZE_MAX)n=std::min(s.size()-so,size_-d);check_hip(hipMemcpyAsync(static_cast<char*>(data_)+d,s.get()+so,n,hipMemcpyHostToDevice,q.stream()),"HIP host-to-device copy");}
    void read(HipQueue&q,VBaseHostBuffer&d,size_t n=SIZE_MAX,size_t so=0,size_t dest=0){if(n==SIZE_MAX)n=std::min(size_-so,d.size()-dest);check_hip(hipMemcpyAsync(d.get()+dest,static_cast<char*>(data_)+so,n,hipMemcpyDeviceToHost,q.stream()),"HIP device-to-host copy");q.wait();}
    void read(HipComputeContext&c,VBaseHostBuffer&d){read(static_cast<HipQueue&>(c),d);}void copy_from(HipQueue&q,const VBaseDeviceBuffer&s,size_t n=SIZE_MAX){if(n==SIZE_MAX)n=std::min(size_,s.size_);check_hip(hipMemcpyAsync(data_,s.data_,n,hipMemcpyDeviceToDevice,q.stream()),"HIP device copy");}void zero(HipQueue&q){check_hip(hipMemsetAsync(data_,0,size_,q.stream()),"hipMemsetAsync");}
private:void*data_=nullptr;size_t size_=0;
};
template<typename T>class VDeviceBuffer:public VBaseDeviceBuffer{public:explicit VDeviceBuffer(VHostBuffer<T>&h):VBaseDeviceBuffer(h.session(),sizeof(T)){}const char*typed_buffer_name()const override{return"VDeviceBuffer";}};
class VDynamicDeviceBuffer:public VBaseDeviceBuffer{public:VDynamicDeviceBuffer(HipSession&s,size_t n):VBaseDeviceBuffer(s,n){}const char*typed_buffer_name()const override{return"VDynamicDeviceBuffer";}};

class HipComputeKernel {
public:HipComputeKernel(HipSession&,const std::string&path,const std::string&symbol);~HipComputeKernel();HipComputeKernel(const HipComputeKernel&)=delete;void launch(HipQueue&,HipDimension grid,HipDimension block,void**args)const;
private:hipModule_t module_=nullptr;hipFunction_t function_=nullptr;
};

namespace rllm::hip_blas {
inline void gemm_abt_half(HipQueue& queue, uint32_t m, uint32_t n, uint32_t k,
                          const VBaseDeviceBuffer& a, const VBaseDeviceBuffer& b,
                          VBaseDeviceBuffer& c) {
    const float alpha=1.0f,beta=0.0f;
    check_hipblas(hipblasGemmEx(queue.blas_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N,
        static_cast<int>(n),static_cast<int>(m),static_cast<int>(k),&alpha,
        b.get(),HIP_R_16F,static_cast<int>(k),
        a.get(),HIP_R_16F,static_cast<int>(k),&beta,
        c.get(),HIP_R_32F,static_cast<int>(n),HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT),"hipblasGemmEx(C = A B^T)");
}
inline void gemm_at_b_acc(HipQueue& queue, uint32_t m, uint32_t n, uint32_t k,
                          const VBaseDeviceBuffer& a, const VBaseDeviceBuffer& b,
                          VBaseDeviceBuffer& c) {
    const float alpha=1.0f,beta=1.0f;
    check_hipblas(hipblasSgemm(queue.blas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_T,
        static_cast<int>(n),static_cast<int>(m),static_cast<int>(k),&alpha,
        static_cast<const float*>(b.get()),static_cast<int>(n),
        static_cast<const float*>(a.get()),static_cast<int>(m),&beta,
        static_cast<float*>(c.get()),static_cast<int>(n)),"hipblasSgemm(C += A^T B)");
}
}
