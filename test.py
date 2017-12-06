#!/usr/bin/python
# coding:utf8
import tensorflow as tf

def test():
	with tf.device("/cpu:0"):
		matrix1 = tf.constant([[3., 3.]])
		matrix2 = tf.constant([[2.],[2.]])
		product = tf.matmul(matrix1, matrix2)
		sess = tf.Session()
		result = sess.run(product)
		print result
		sess.close()

def test2():
	sess = tf.InteractiveSession()

	x = tf.Variable([1.0, 2.0])
	a = tf.constant([3.0, 3.0])
	x.initializer.run()
	sub = tf.sub(x,a)
	print sub.eval()

def test3():
	# 创建一个变量, 初始化为标量 0.
	state = tf.Variable(0, name="counter")

	# 创建一个 op, 其作用是使 state 增加 1

	one = tf.constant(1)
	two = tf.constant(2)
	new_value = tf.add(state, one)
	my = tf.add(state,two)
	update = tf.assign(state, new_value)
	update1 = tf.assign(state,my)
	# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
	# 首先必须增加一个`初始化` op 到图中.
	init_op = tf.initialize_all_variables()

	# 启动图, 运行 op
	with tf.Session() as sess:
	  # 运行 'init' op
	  sess.run(init_op)
	  # 打印 'state' 的初始值
	  print sess.run(state)
	  # 运行 op, 更新 'state', 并打印 'state'
	  for _ in range(3):
	    print sess.run(update1)
	    # print sess.run(state)
