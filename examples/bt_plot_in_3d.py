import cv2
import time
import pykinect_azure as pykinect
import matplotlib.pyplot as plt
from pykinect_azure.k4abt._k4abtTypes import K4ABT_JOINT_NAMES, K4ABT_SEGMENT_PAIRS
from scipy.spatial.transform import Rotation as R

def plot_in_3d(ax, joints):
	# Plot the joints
	for i in range(len(joints)):
		# if K4ABT_JOINT_NAMES[i] not in ['left handtip', 'right handtip', 'left thumb', 'right thumb']:
		# 	ax.scatter(joints[i][0], joints[i][1], joints[i][2], c='r', marker='o')
		
		if K4ABT_JOINT_NAMES[i] in ["left hand", "right hand"]:
			quat = joints[i][3:7]
			r = R.from_quat(quat).as_matrix()

			ax.scatter(joints[i][0], joints[i][1], joints[i][2], c='r', marker='o')
			ax.text(joints[i][0], joints[i][1], joints[i][2], K4ABT_JOINT_NAMES[i])
			ax.quiver(joints[i][0], joints[i][1], joints[i][2], r[0, 0], r[1, 0], r[2, 0], length=150, color='r')
			ax.quiver(joints[i][0], joints[i][1], joints[i][2], r[0, 1], r[1, 1], r[2, 1], length=150, color='g')
			ax.quiver(joints[i][0], joints[i][1], joints[i][2], r[0, 2], r[1, 2], r[2, 2], length=150, color='b')

	# Plot the segments
	# for i in range(len(K4ABT_SEGMENT_PAIRS)):
	# 	joint1 = joints[K4ABT_SEGMENT_PAIRS[i][0]]
	# 	joint2 = joints[K4ABT_SEGMENT_PAIRS[i][1]]
	# 	if K4ABT_JOINT_NAMES[K4ABT_SEGMENT_PAIRS[i][0]] in ['left handtip', 'right handtip', 'left thumb', 'right thumb'] or \
	# 		K4ABT_JOINT_NAMES[K4ABT_SEGMENT_PAIRS[i][1]] in ['left handtip', 'right handtip', 'left thumb', 'right thumb']:
	# 		continue
	# 	ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], c='b')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim3d(-500, 500)
	ax.set_ylim3d(0, 1000)
	ax.set_zlim3d(0, 1000)
	plt.pause(0.001)
	plt.cla()


if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
	device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5
	#print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker()

	cv2.namedWindow('Color image with skeleton',cv2.WINDOW_NORMAL)
	ax = plt.axes(projection='3d')
	while True:
		start = time.time()
		
		# Get capture
		capture = device.update()

		# Get body tracker frame
		body_frame = bodyTracker.update()

		# Get the color image
		ret, color_image = capture.get_color_image()

		if not ret:
			continue

		# Draw the skeletons into the color image
		body = body_frame.get_bodies()
		if len(body) > 0:
			# print(body[0])
			plot_in_3d(ax, body[0].numpy())
		
		color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
		# Overlay body segmentation on depth image
		cv2.imshow('Color image with skeleton',color_skeleton)	

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):  
			break

		print(f'FPS: {1/(time.time()-start)}')