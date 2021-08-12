import numpy as np
import rosbag

def load_heightmapgrid_from_bag(filename, topicname="/heightmap"):
    bag = rosbag.Bag(filename)
    topic, msg, t  = next( bag.read_messages(topics=[topicname]))
    bins = np.asarray(msg.bins)
    return bins[np.asarray(msg.data)].reshape(msg.info.height, msg.info.width)

def visualize_elevationmap(img, filename):
    img /= np.max(img)
    img *= 255
    img = img.astype(np.int8)
    Image.fromarray(img).save(filename)
