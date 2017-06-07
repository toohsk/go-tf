package main

import (
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	root := op.NewScope()

	A := op.Placeholder(root.SubScope("input"), tf.Int32, op.PlaceholderShape(tf.MakeShape(2, 2)))
	X := op.Placeholder(root.SubScope("input"), tf.Int32, op.PlaceholderShape(tf.MakeShape(2, 1)))
	fmt.Println(A.Op.Name(), X.Op.Name())

	product := op.MatMul(root, A, X)

	graph, err := root.Finalize()
	if err != nil {
		panic(err.Error())
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		panic(err.Error())
	}

	var matrix, column *tf.Tensor

	if matrix, err = tf.NewTensor([2][2]int32{{1, 2}, {-1, -2}}); err != nil {
		panic(err.Error())
	}

	if column, err = tf.NewTensor([2][1]int32{{10}, {100}}); err != nil {
		panic(err.Error())
	}

	var results []*tf.Tensor
	if results, err = sess.Run(map[tf.Output]*tf.Tensor{
		A: matrix,
		X: column,
	}, []tf.Output{product}, nil); err != nil {
		panic(err.Error())
	}

	for _, result := range results {
		fmt.Println(result.Value().([][]int32))
	}
}
