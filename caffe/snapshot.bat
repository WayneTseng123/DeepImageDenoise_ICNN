cd ../../../
"Build/x64/Release/caffe.exe" train  --solver=examples/mylenet_denoise/3579-3-3-3-3-3-3-3/multinet_solver.prototxt -snapshot examples/mylenet_denoise/3579-3-3-3-3-3-3-3/residual_model/multinet_iter_100000.solverstate
pause