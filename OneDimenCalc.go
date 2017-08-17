package main

import (
	"fmt"

	"github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	root := op.NewScope()
	A := op.Placeholder(root.SubScope("input"), tensorflow.Int32, op.PlaceholderShape(tensorflow.MakeShape(1, 1)))
	x := op.Placeholder(root.SubScope("input"), tensorflow.Int32, op.PlaceholderShape(tensorflow.MakeShape(1, 1)))
	product := op.MatMul(root, A, x)

	graph, err := root.Finalize()
	if err != nil {
		panic(err.Error())
	}
	var sess *tensorflow.Session
	sess, err = tensorflow.NewSession(graph, &tensorflow.SessionOptions{})
	defer sess.Close()
	if err != nil {
		panic(err.Error())
	}

	var matrix1, matrix2 *tensorflow.Tensor

	if matrix1, err = tensorflow.NewTensor([1][1]int32{{10}}); err != nil {
		panic(err.Error())
	}
	if matrix2, err = tensorflow.NewTensor([1][1]int32{{120}}); err != nil {
		panic(err.Error())
	}

	var results []*tensorflow.Tensor
	if results, err = sess.Run(map[tensorflow.Output]*tensorflow.Tensor{
		A: matrix1,
		x: matrix2,
	}, []tensorflow.Output{product}, nil); err != nil {
		panic(err.Error())
	}
	for _, result := range results {
		fmt.Println(result.Value().([][]int32))
	}

}
