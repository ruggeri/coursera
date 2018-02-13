import config
import graph as g
import tensorflow as tf
import train

def maybe_restore(session, saver):
    # Was there a reason this was ever here??
    # saver = tf.train.Saver()
    while True:
        ipt = input("restore [y/n]: ")
        if ipt == "y":
            print("Restoring!")
            saver.restore(session, config.CHECKPOINT_FILENAME)
            break
        elif ipt == "n":
            print("Starting fresh!")
            session.run(tf.global_variables_initializer())
            break

if __name__ == "__main__":
    with tf.Session() as session:
        graph = g.build_graph()
        saver = tf.train.Saver()

        maybe_restore(session, saver)
        train.train(session, graph, saver)
