package cudamatrix

import "C"

// Source: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
//         #group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038

const (
	// CudaSuccess - The API call returned with no errors. In the case of query calls, this also means that the
	// operation being queried is complete (see cudaEventQuery() and cudaStreamQuery()).
	CudaSuccess C.int = 0

	// CudaErrorInvalidValue - This indicates that one or more of the parameters passed to the API call is not
	// within an acceptable range of values.
	CudaErrorInvalidValue C.int = 1

	// CudaErrorMemoryAllocation - The API call failed because it was unable to allocate enough memory or other
	// resources to perform the requested operation.
	CudaErrorMemoryAllocation C.int = 2

	// CudaErrorInitializationError - The API call failed because the CUDA driver and runtime could not be
	// initialized.
	CudaErrorInitializationError C.int = 3

	// CudaErrorCudartUnloading - This indicates that a CUDA Runtime API call cannot be executed because it
	// is being called during process shut down, at a point in time after CUDA driver has been unloaded.
	CudaErrorCudartUnloading C.int = 4

	// CudaErrorProfilerDisabled - This indicates profiler is not initialized for this run. This can happen
	// when the application is running with external profiling tools like visual profiler.
	CudaErrorProfilerDisabled C.int = 5

	// CudaErrorProfilerNotInitialized - (Deprecated) This error return is deprecated as of CUDA 5.0. It is no
	// longer an error to attempt to enable/disable the profiling via cudaProfilerStart or cudaProfilerStop
	// without initialization.
	CudaErrorProfilerNotInitialized C.int = 6

	// CudaErrorProfilerAlreadyStarted - (Deprecated) This error return is deprecated as of CUDA 5.0.It is no
	// longer an error to call cudaProfilerStart() when profiling is already enabled.
	CudaErrorProfilerAlreadyStarted C.int = 7

	// CudaErrorProfilerAlreadyStopped - (Deprecated) This error return is deprecated as of CUDA 5.0.It is no
	// longer an error to call cudaProfilerStop() when profiling is already disabled.
	CudaErrorProfilerAlreadyStopped C.int = 8

	// CudaErrorInvalidConfiguration - This indicates that a kernel launch is requesting resources that can
	// never be satisfied by the current device.Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks.See cudaDeviceProp for more device limitations.
	CudaErrorInvalidConfiguration C.int = 9

	// CudaErrorInvalidPitchValue - This indicates that one or more of the pitch-related parameters passed to
	// the API call is not within the acceptable range for pitch.
	CudaErrorInvalidPitchValue C.int = 12

	// CudaErrorInvalidSymbol - This indicates that the symbol name/identifier passed to the API call is not a
	// valid name or identifier.
	CudaErrorInvalidSymbol C.int = 13

	// CudaErrorInvalidHostPointer - (Deprecated) This error return is deprecated as of CUDA 10.1.
	// This indicates that at least one host pointer passed to the API call is not a valid host pointer.
	CudaErrorInvalidHostPointer C.int = 16

	// CudaErrorInvalidDevicePointer - (Deprecated) This error return is deprecated as of CUDA 10.1.
	// This indicates that at least one device pointer passed to the API call is not a valid device pointer.
	CudaErrorInvalidDevicePointer C.int = 17

	// CudaErrorInvalidTexture - This indicates that the texture passed to the API call is not a valid texture.
	CudaErrorInvalidTexture C.int = 18

	// CudaErrorInvalidTextureBinding - This indicates that the texture binding is not valid.This occurs if you call
	// cudaGetTextureAlignmentOffset() with an unbound texture.
	CudaErrorInvalidTextureBinding C.int = 19

	// CudaErrorInvalidChannelDescriptor - This indicates that the channel descriptor passed to the API call is not
	// valid.This occurs if the format is not one of the formats specified by cudaChannelFormatKind, or if one of
	// the dimensions is invalid.
	CudaErrorInvalidChannelDescriptor C.int = 20

	// CudaErrorInvalidMemcpyDirection - This indicates that the direction of the memcpy passed to the API call is
	// not one of the types specified by cudaMemcpyKind.
	CudaErrorInvalidMemcpyDirection C.int = 21

	// CudaErrorAddressOfConstant - (Deprecated) This error return is deprecated as of CUDA 3.1.Variables in
	// constant memory may now have their address taken by the runtime via cudaGetSymbolAddress(). This indicated
	// that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release.
	CudaErrorAddressOfConstant C.int = 22

	// CudaErrorTextureFetchFailed - (Deprecated) This error return is deprecated as of CUDA 3.1. Device emulation
	// mode was removed with the CUDA 3.1 release. This indicated that a texture fetch was not able to be performed.
	// This was previously used for device emulation of texture operations.
	CudaErrorTextureFetchFailed C.int = 23

	// CudaErrorTextureNotBound - (Deprecated) This error return is deprecated as of CUDA 3.1.Device emulation mode
	// was removed with the CUDA 3.1 release. This indicated that a texture was not bound for access.This was
	// previously used for device emulation of texture operations.
	CudaErrorTextureNotBound C.int = 24

	// CudaErrorSynchronizationError - (Deprecated) This error return is deprecated as of CUDA 3.1. Device emulation
	// mode was removed with the CUDA 3.1 release. This indicated that a synchronization operation had failed. This
	// was previously used for some device emulation functions.
	CudaErrorSynchronizationError C.int = 25

	// CudaErrorInvalidFilterSetting - This indicates that a non-float texture was being accessed with linear
	// filtering.This is not supported by CUDA.
	CudaErrorInvalidFilterSetting C.int = 26

	// CudaErrorInvalidNormSetting - This indicates that an attempt was made to read a non-float texture as a
	// normalized float.This is not supported by CUDA.
	CudaErrorInvalidNormSetting C.int = 27

	// CudaErrorMixedDeviceExecution - (Deprecated) This error return is deprecated as of CUDA 3.1.Device emulation
	// mode was removed with the CUDA 3.1 release. Mixing of device and device emulation code was not allowed.
	CudaErrorMixedDeviceExecution C.int = 28

	// CudaErrorNotYetImplemented - (Deprecated) This error return is deprecated as of CUDA 4.1.
	// This indicates that the API call is not yet implemented.Production releases of CUDA will never return this error.
	CudaErrorNotYetImplemented C.int = 31

	// CudaErrorMemoryValueTooLarge - (Deprecated) This error return is deprecated as of CUDA 3.1.Device emulation
	// mode was removed with the CUDA 3.1 release. This indicated that an emulated device pointer exceeded the
	// 32-bit address range.
	CudaErrorMemoryValueTooLarge C.int = 32

	// CudaErrorStubLibrary - This indicates that the CUDA driver that the application has loaded is a stub
	// library.Applications that run with the stub rather than a real driver loaded will result in CUDA API
	// returning this error.
	CudaErrorStubLibrary C.int = 34

	// CudaErrorInsufficientDriver - This indicates that the installed NVIDIA CUDA driver is older than the
	// CUDA runtime library.This is not a supported configuration.Users should install an updated NVIDIA display
	// driver to allow the application to run.
	CudaErrorInsufficientDriver C.int = 35

	// CudaErrorCallRequiresNewerDriver - This indicates that the API call requires a newer CUDA driver than the one
	// currently installed.Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.
	CudaErrorCallRequiresNewerDriver C.int = 36

	// CudaErrorInvalidSurface - This indicates that the surface passed to the API call is not a valid surface.
	CudaErrorInvalidSurface C.int = 37

	// CudaErrorDuplicateVariableName - This indicates that multiple global or constant variables (across separate
	// CUDA source files in the application) share the same string name.
	CudaErrorDuplicateVariableName C.int = 43

	// CudaErrorDuplicateTextureName - This indicates that multiple textures (across separate CUDA source files in
	// the application) share the same string name.
	CudaErrorDuplicateTextureName C.int = 44

	// CudaErrorDuplicateSurfaceName - This indicates that multiple surfaces (across separate CUDA source files
	// in the application) share the same string name.
	CudaErrorDuplicateSurfaceName C.int = 45

	// CudaErrorDevicesUnavailable - This indicates that all CUDA devices are busy or unavailable at the current
	// time. Devices are often busy/unavailable due to use of cudaComputeModeProhibited,
	// cudaComputeModeExclusiveProcess, or when long-running CUDA kernels have filled up the GPU and are blocking
	// new work from starting.They can also be unavailable due to memory constraints on a device that already has
	// active CUDA work being performed.
	CudaErrorDevicesUnavailable C.int = 46

	// CudaErrorIncompatibleDriverContext - This indicates that the current context is not compatible with this
	// the CUDA Runtime.This can only occur if you are using CUDA Runtime/Driver interoperability and have created an
	// existing Driver context using the driver API.The Driver context may be incompatible either because the Driver
	// context was created using an older version of the API, because the Runtime API call expects a primary driver
	// context and the Driver context is not primary, or because the Driver context has been destroyed.Please see
	// "Interactions with the CUDA Driver API" for more information.
	CudaErrorIncompatibleDriverContext C.int = 49

	// CudaErrorMissingConfiguration - The device function being invoked (usually via cudaLaunchKernel()) was
	// not previously configured via the cudaConfigureCall() function.
	CudaErrorMissingConfiguration C.int = 52

	// CudaErrorPriorLaunchFailure - (Deprecated) This error return is deprecated as of CUDA 3.1.Device emulation
	// mode was removed with the CUDA 3.1 release. This indicated that a previous kernel launch failed.This was
	// previously used for device emulation of kernel launches.
	CudaErrorPriorLaunchFailure C.int = 53

	// CudaErrorLaunchMaxDepthExceeded - This error indicates that a device runtime grid launch did not occur
	// because the depth of the child grid would exceed the maximum supported number of nested grid launches.
	CudaErrorLaunchMaxDepthExceeded C.int = 65

	// CudaErrorLaunchFileScopedTex - This error indicates that a grid launch did not occur because the kernel
	// uses file-scoped textures which are unsupported by the device runtime.Kernels launched via the device
	// runtime only support textures created with the Texture Object APIs.
	CudaErrorLaunchFileScopedTex C.int = 66

	// CudaErrorLaunchFileScopedSurf - This error indicates that a grid launch did not occur because the kernel
	// uses file-scoped surfaces which are unsupported by the device runtime.Kernels launched via the device
	// runtime only support surfaces created with the Surface Object APIs.
	CudaErrorLaunchFileScopedSurf C.int = 67

	// CudaErrorSyncDepthExceeded - This error indicates that a call to cudaDeviceSynchronize made from the device
	// runtime failed because the call was made at grid depth greater than either the default (2 levels of
	// grids) or user specified device limit cudaLimitDevRuntimeSyncDepth.To be able to synchronize on launched
	// grids at a greater depth successfully, the maximum nested depth at which cudaDeviceSynchronize will be
	// called must be specified with the cudaLimitDevRuntimeSyncDepth limit to the cudaDeviceSetLimit api before
	// the host-side launch of a kernel using the device runtime.Keep in mind that additional levels of sync depth
	// require the runtime to reserve large amounts of device memory that cannot be used for user allocations.
	// Note that cudaDeviceSynchronize made from device runtime is only supported on devices of compute
	// capability < 9.0.
	CudaErrorSyncDepthExceeded C.int = 68

	// CudaErrorLaunchPendingCountExceeded - This error indicates that a device runtime grid launch failed because
	// the launch would exceed the limit cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed
	// successfully, cudaDeviceSetLimit must be called to set the cudaLimitDevRuntimePendingLaunchCount to be
	// higher than the upper bound of outstanding launches that can be issued to the device runtime.Keep in mind
	// that raising the limit of pending device runtime launches will require the runtime to reserve device memory
	// that cannot be used for user allocations.
	CudaErrorLaunchPendingCountExceeded C.int = 69

	// CudaErrorInvalidDeviceFunction - The requested device function does not exist or is not compiled for the
	// proper device architecture.
	CudaErrorInvalidDeviceFunction C.int = 98

	// CudaErrorNoDevice - This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
	CudaErrorNoDevice C.int = 100

	// CudaErrorInvalidDevice - This indicates that the device ordinal supplied by the user does not correspond
	// to a valid CUDA device or that the action requested is invalid for the specified device.
	CudaErrorInvalidDevice C.int = 101

	// CudaErrorDeviceNotLicensed - This indicates that the device doesn't have a valid Grid License.
	CudaErrorDeviceNotLicensed C.int = 102

	// CudaErrorSoftwareValidityNotEstablished - By default, the CUDA runtime may perform a minimal set of
	// self-tests, as well as CUDA driver tests, to establish the validity of both.Introduced in CUDA 11.2,
	// this error return indicates that at least one of these tests has failed and the validity of either the
	// runtime or the driver could not be established.
	CudaErrorSoftwareValidityNotEstablished C.int = 103

	// CudaErrorStartupFailure - This indicates an internal startup failure in the CUDA runtime.
	CudaErrorStartupFailure C.int = 127

	// CudaErrorInvalidKernelImage - This indicates that the device kernel image is invalid.
	CudaErrorInvalidKernelImage C.int = 200

	// CudaErrorDeviceUninitialized - This most frequently indicates that there is no context bound to the current
	// thread.This can also be returned if the context passed to an API call is not a valid handle (such as a
	// context that has had cuCtxDestroy() invoked on it).This can also be returned if a user mixes different
	// API versions (i.e.3010 context with 3020 API calls).See cuCtxGetApiVersion() for more details.
	CudaErrorDeviceUninitialized C.int = 201

	// CudaErrorMapBufferObjectFailed - This indicates that the buffer object could not be mapped.
	CudaErrorMapBufferObjectFailed C.int = 205

	// CudaErrorUnmapBufferObjectFailed - This indicates that the buffer object could not be unmapped.
	CudaErrorUnmapBufferObjectFailed C.int = 206

	// CudaErrorArrayIsMapped - This indicates that the specified array is currently mapped and thus
	// cannot be destroyed.
	CudaErrorArrayIsMapped C.int = 207

	// CudaErrorAlreadyMapped - This indicates that the resource is already mapped.
	CudaErrorAlreadyMapped C.int = 208

	// CudaErrorNoKernelImageForDevice - This indicates that there is no kernel image available that is suitable
	// for the device.This can occur when a user specifies code generation options for a particular CUDA source
	// file that do not include the corresponding device configuration.
	CudaErrorNoKernelImageForDevice C.int = 209

	// CudaErrorAlreadyAcquired - This indicates that a resource has already been acquired.
	CudaErrorAlreadyAcquired C.int = 210

	// CudaErrorNotMapped - This indicates that a resource is not mapped.
	CudaErrorNotMapped C.int = 211

	// CudaErrorNotMappedAsArray - This indicates that a mapped resource is not available for access as an array.
	CudaErrorNotMappedAsArray C.int = 212

	// CudaErrorNotMappedAsPointer - This indicates that a mapped resource is not available for access as a pointer.
	CudaErrorNotMappedAsPointer C.int = 213

	// CudaErrorECCUncorrectable - This indicates that an uncorrectable ECC error was detected during execution.
	CudaErrorECCUncorrectable C.int = 214

	// CudaErrorUnsupportedLimit - This indicates that the cudaLimit passed to the API call is not supported
	// by the active device.
	CudaErrorUnsupportedLimit C.int = 215

	// CudaErrorDeviceAlreadyInUse - This indicates that a call tried to access an exclusive-thread device that is
	// already in use by a different thread.
	CudaErrorDeviceAlreadyInUse C.int = 216

	// CudaErrorPeerAccessUnsupported - This error indicates that P2P access is not supported across the given devices.
	CudaErrorPeerAccessUnsupported C.int = 217

	// CudaErrorInvalidPtx - A PTX compilation failed.The runtime may fall back to compiling PTX if an application
	// does not contain a suitable binary for the current device.
	CudaErrorInvalidPtx C.int = 218

	// CudaErrorInvalidGraphicsContext - This indicates an error with the OpenGL or DirectX context.
	CudaErrorInvalidGraphicsContext C.int = 219

	// CudaErrorNvlinkUncorrectable - This indicates that an uncorrectable NVLink error was detected during
	// the execution.
	CudaErrorNvlinkUncorrectable C.int = 220

	// CudaErrorJitCompilerNotFound - This indicates that the PTX JIT compiler library was not found.The JIT Compiler
	// library is used for PTX compilation.The runtime may fall back to compiling PTX if an application does not
	// contain a suitable binary for the current device.
	CudaErrorJitCompilerNotFound C.int = 221

	// CudaErrorUnsupportedPtxVersion - This indicates that the provided PTX was compiled with an unsupported
	// toolchain.The most common reason for this, is the PTX was generated by a compiler newer than what is
	// supported by the CUDA driver and PTX JIT compiler.
	CudaErrorUnsupportedPtxVersion C.int = 222

	// CudaErrorJitCompilationDisabled - This indicates that the JIT compilation was disabled.The JIT compilation
	// compiles PTX.The runtime may fall back to compiling PTX if an application does not contain a suitable
	// binary for the current device.
	CudaErrorJitCompilationDisabled C.int = 223

	// CudaErrorUnsupportedExecAffinity - This indicates that the provided execution affinity is not supported
	// by the device.
	CudaErrorUnsupportedExecAffinity C.int = 224

	// CudaErrorUnsupportedDevSideSync - This indicates that the code to be compiled by the PTX JIT contains
	// unsupported call to cudaDeviceSynchronize.
	CudaErrorUnsupportedDevSideSync C.int = 225

	// CudaErrorInvalidSource - This indicates that the device kernel source is invalid.
	CudaErrorInvalidSource C.int = 300

	// CudaErrorFileNotFound - This indicates that the file specified was not found.
	CudaErrorFileNotFound C.int = 301

	// CudaErrorSharedObjectSymbolNotFound - This indicates that a link to a shared object failed to resolve.
	CudaErrorSharedObjectSymbolNotFound C.int = 302

	// CudaErrorSharedObjectInitFailed - This indicates that initialization of a shared object failed.
	CudaErrorSharedObjectInitFailed C.int = 303

	// CudaErrorOperatingSystem - This error indicates that an OS call failed.
	CudaErrorOperatingSystem C.int = 304

	// CudaErrorInvalidResourceHandle - This indicates that a resource handle passed to the API call was not valid.
	// Resource handles are opaque types like cudaStream_t and cudaEvent_t.
	CudaErrorInvalidResourceHandle C.int = 400

	// CudaErrorIllegalState - This indicates that a resource required by the API call is not in a valid state to
	// perform the requested operation.
	CudaErrorIllegalState C.int = 401

	// CudaErrorLossyQuery - This indicates an attempt was made to introspect an object in a way that would
	// discard semantically important information.This is either due to the object using functionality newer than
	// the API version used to introspect it or omission of optional return arguments.
	CudaErrorLossyQuery C.int = 402

	// CudaErrorSymbolNotFound - This indicates that a named symbol was not found.Examples of symbols are
	// global/constant variable names, driver function names, texture names, and surface names.
	CudaErrorSymbolNotFound C.int = 500

	// CudaErrorNotReady - This indicates that asynchronous operations issued previously have not completed yet.
	// This result is not actually an error, but must be indicated differently than cudaSuccess (which indicates
	// completion).Calls that may return this value include cudaEventQuery() and cudaStreamQuery().
	CudaErrorNotReady C.int = 600

	// CudaErrorIllegalAddress - The device encountered a load or store instruction on an invalid memory address.
	// This leaves the process in an inconsistent state and any further CUDA work will return the same error.
	// To continue using CUDA, the process must be terminated and relaunched.
	CudaErrorIllegalAddress C.int = 700

	// CudaErrorLaunchOutOfResources - This indicates that a launch did not occur because it did not have
	// appropriate resources.Although this error is similar to CudaErrorInvalidConfiguration, this error usually
	// indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel
	// launch specifies too many threads for the kernel's register count.
	CudaErrorLaunchOutOfResources C.int = 701

	// CudaErrorLaunchTimeout - This indicates that the device kernel took too long to execute.This can only
	// occur if timeouts are enabled - see the device property kernelExecTimeoutEnabled for more information.
	// This leaves the process in an inconsistent state and any further CUDA work will return the same error.
	// To continue using CUDA, the process must be terminated and relaunched.
	CudaErrorLaunchTimeout C.int = 702

	// CudaErrorLaunchIncompatibleTexturing - This error indicates a kernel launch that uses an incompatible
	// texturing mode.
	CudaErrorLaunchIncompatibleTexturing C.int = 703

	// CudaErrorPeerAccessAlreadyEnabled - This error indicates that a call to cudaDeviceEnablePeerAccess()
	// is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.
	CudaErrorPeerAccessAlreadyEnabled C.int = 704

	// CudaErrorPeerAccessNotEnabled - This error indicates that cudaDeviceDisablePeerAccess() is trying to disable
	// peer addressing which has not been enabled yet via cudaDeviceEnablePeerAccess().
	CudaErrorPeerAccessNotEnabled C.int = 705

	// CudaErrorSetOnActiveProcess - This indicates that the user has called cudaSetValidDevices(),
	// cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(),
	// or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations
	// (allocating memory and launching kernels are examples of non-device management operations).This error can
	// also be returned if using runtime/driver interoperability and there is an existing CUcontext active on the
	// host thread.
	CudaErrorSetOnActiveProcess C.int = 708

	// CudaErrorContextIsDestroyed - This error indicates that the context current to the calling thread has been
	// destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.
	CudaErrorContextIsDestroyed C.int = 709

	// CudaErrorAssert - An assert triggered in device code during kernel execution.The device cannot be used
	// again. All existing allocations are invalid.To continue using CUDA, the process must be terminated and
	// relaunched.
	CudaErrorAssert C.int = 710

	// CudaErrorTooManyPeers - This error indicates that the hardware resources required to enable peer access
	// have been exhausted for one or more of the devices passed to cudaEnablePeerAccess().
	CudaErrorTooManyPeers C.int = 711

	// CudaErrorHostMemoryAlreadyRegistered - This error indicates that the memory range passed to cudaHostRegister()
	// has already been registered.
	CudaErrorHostMemoryAlreadyRegistered C.int = 712

	// CudaErrorHostMemoryNotRegistered - This error indicates that the pointer passed to cudaHostUnregister() does
	// not correspond to any currently registered memory region.
	CudaErrorHostMemoryNotRegistered C.int = 713

	// CudaErrorHardwareStackError - Device encountered an error in the call stack during kernel execution,
	// possibly due to stack corruption or exceeding the stack size limit.This leaves the process in an
	// inconsistent state and any further CUDA work will return the same error.To continue using CUDA, the
	// process must be terminated and relaunched.
	CudaErrorHardwareStackError C.int = 714

	// CudaErrorIllegalInstruction - The device encountered an illegal instruction during kernel execution.
	// This leaves the process in an inconsistent state and any further CUDA work will return the same error.
	// To continue using CUDA, the process must be terminated and relaunched.
	CudaErrorIllegalInstruction C.int = 715

	// CudaErrorMisalignedAddress - The device encountered a load or store instruction on a memory address which
	// is not aligned.This leaves the process in an inconsistent state and any further CUDA work will return the
	// same error.To continue using CUDA, the process must be terminated and relaunched.
	CudaErrorMisalignedAddress C.int = 716

	// CudaErrorInvalidAddressSpace - While executing a kernel, the device encountered an instruction which can
	// only operate on memory locations in certain address spaces (global, shared, or local), but was supplied
	// a memory address not belonging to an allowed address space.This leaves the process in an inconsistent
	// state and any further CUDA work will return the same error.To continue using CUDA, the process must
	// be terminated and relaunched.
	CudaErrorInvalidAddressSpace C.int = 717

	// CudaErrorInvalidPc - The device encountered an invalid program counter.This leaves the process in an
	// inconsistent state and any further CUDA work will return the same error.To continue using CUDA, the process
	// must be terminated and relaunched.
	CudaErrorInvalidPc C.int = 718

	// CudaErrorLaunchFailure - An exception occurred on the device while executing a kernel.Common causes include
	// dereferencing an invalid device pointer and accessing out of bounds shared memory.Less common cases can be
	// system specific - more information about these cases can be found in the system specific user guide. This
	// leaves the process in an inconsistent state and any further CUDA work will return the same error.To continue
	// using CUDA, the process must be terminated and relaunched.
	CudaErrorLaunchFailure C.int = 719

	// CudaErrorCooperativeLaunchTooLarge - This error indicates that the number of blocks launched per grid for a
	// kernel that was launched via either cudaLaunchCooperativeKernel or cudaLaunchCooperativeKernelMultiDevice
	// exceeds the maximum number of blocks as allowed by cudaOccupancyMaxActiveBlocksPerMultiprocessor or
	// cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified
	// by the device attribute cudaDevAttrMultiProcessorCount.
	CudaErrorCooperativeLaunchTooLarge C.int = 720

	// CudaErrorNotPermitted - This error indicates the attempted operation is not permitted.
	CudaErrorNotPermitted C.int = 800

	// CudaErrorNotSupported - This error indicates the attempted operation is not supported on the
	// current system or device.
	CudaErrorNotSupported C.int = 801

	// CudaErrorSystemNotReady - This error indicates that the system is not yet ready to start any CUDA work.
	// To continue using CUDA, verify the system configuration is in a valid state and all required driver
	// daemons are actively running.More information about this error can be found in the system specific user guide.
	CudaErrorSystemNotReady C.int = 802

	// CudaErrorSystemDriverMismatch - This error indicates that there is a mismatch between the versions of the
	// display driver and the CUDA driver.Refer to the compatibility documentation for supported versions.
	CudaErrorSystemDriverMismatch C.int = 803

	// CudaErrorCompatNotSupportedOnDevice - This error indicates that the system was upgraded to run with forward
	// compatibility but the visible hardware detected by CUDA does not support this configuration.Refer to the
	// compatibility documentation for the supported hardware matrix or ensure that only supported hardware is
	// visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
	CudaErrorCompatNotSupportedOnDevice C.int = 804

	// CudaErrorMpsConnectionFailed - This error indicates that the MPS client failed to connect to the MPS
	// control daemon or the MPS server.
	CudaErrorMpsConnectionFailed C.int = 805

	// CudaErrorMpsRpcFailure - This error indicates that the remote procedural call between the MPS server
	// and the MPS client failed.
	CudaErrorMpsRpcFailure C.int = 806

	// CudaErrorMpsServerNotReady - This error indicates that the MPS server is not ready to accept new MPS
	// client requests.This error can be returned when the MPS server is in the process of recovering from
	// a fatal failure.
	CudaErrorMpsServerNotReady C.int = 807

	// CudaErrorMpsMaxClientsReached - This error indicates that the hardware resources required to create
	// MPS client have been exhausted.
	CudaErrorMpsMaxClientsReached C.int = 808

	// CudaErrorMpsMaxConnectionsReached - This error indicates the the hardware resources required to device
	// connections have been exhausted.
	CudaErrorMpsMaxConnectionsReached C.int = 809

	// CudaErrorMpsClientTerminated - This error indicates that the MPS client has been terminated by the server.
	// To continue using CUDA, the process must be terminated and relaunched.
	CudaErrorMpsClientTerminated C.int = 810

	// CudaErrorCdpNotSupported - This error indicates, that the program is using CUDA Dynamic Parallelism,
	// but the current configuration, like MPS, does not support it.
	CudaErrorCdpNotSupported C.int = 811

	// CudaErrorCdpVersionMismatch - This error indicates, that the program contains an unsupported interaction
	// between different versions of CUDA Dynamic Parallelism.
	CudaErrorCdpVersionMismatch C.int = 812

	// CudaErrorStreamCaptureUnsupported - The operation is not permitted when the stream is capturing.
	CudaErrorStreamCaptureUnsupported C.int = 900

	// CudaErrorStreamCaptureInvalidated - The current capture sequence on the stream has been invalidated
	// due to a previous error.
	CudaErrorStreamCaptureInvalidated C.int = 901

	// CudaErrorStreamCaptureMerge - The operation would have resulted in a merge of two independent capture sequences.
	CudaErrorStreamCaptureMerge C.int = 902

	// CudaErrorStreamCaptureUnmatched - The capture was not initiated in this stream.
	CudaErrorStreamCaptureUnmatched C.int = 903

	// CudaErrorStreamCaptureUnjoined - The capture sequence contains a fork that was not joined to the primary stream.
	CudaErrorStreamCaptureUnjoined C.int = 904

	// CudaErrorStreamCaptureIsolation - A dependency would have been created which crosses the capture sequence
	// boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
	CudaErrorStreamCaptureIsolation C.int = 905

	// CudaErrorStreamCaptureImplicit - The operation would have resulted in a disallowed implicit dependency on
	// a current capture sequence from cudaStreamLegacy.
	CudaErrorStreamCaptureImplicit C.int = 906

	// CudaErrorCapturedEvent - The operation is not permitted on an event which was last recorded in a capturing
	// stream.
	CudaErrorCapturedEvent C.int = 907

	// CudaErrorStreamCaptureWrongThread - A stream capture sequence not initiated with the
	// cudaStreamCaptureModeRelaxed argument to cudaStreamBeginCapture was passed to cudaStreamEndCapture
	// in a different thread.
	CudaErrorStreamCaptureWrongThread C.int = 908

	// CudaErrorTimeout - This indicates that the wait operation has timed out.
	CudaErrorTimeout C.int = 909

	// CudaErrorGraphExecUpdateFailure - This error indicates that the graph update was not performed because
	// it included changes which violated constraints specific to instantiated graph update.
	CudaErrorGraphExecUpdateFailure C.int = 910

	// CudaErrorExternalDevice - This indicates that an async error has occurred in a device outside of CUDA.
	// If CUDA was waiting for an external device's signal before consuming shared data, the external device
	// signaled an error indicating that the data is not valid for consumption. This leaves the process in an
	// inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the
	// process must be terminated and relaunched.
	CudaErrorExternalDevice C.int = 911

	// CudaErrorInvalidClusterSize - This indicates that a kernel launch error has occurred due to cluster
	// misconfiguration.
	CudaErrorInvalidClusterSize C.int = 912

	// CudaErrorUnknown - This indicates that an unknown internal error has occurred.
	CudaErrorUnknown C.int = 999

	// CudaErrorApiFailureBase - error is not defined in nvidia docs
	CudaErrorApiFailureBase C.int = 10000
)
