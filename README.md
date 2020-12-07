# distributed_model_evalulation

Before running please install [keras-retinanet](https://github.com/fizyr/keras-retinanet)

In order to change the number of workers (number of models to load in) change `numNodes` in hub.go

To run
```
go run hub.go <video_path> <model_path>
```
