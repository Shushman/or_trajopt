#!//bin/env python
from __future__ import print_function
import numpy
import unittest
import openravepy
import subprocess
import os
import sys

from or_trajopt import TrajoptPlanner


# Add the models included with OpenRAVE to the OPENRAVE_DATA path. These may
# not be available if the user manually set the OPENRAVE_DATA environmental
# variable, e.g. through openrave_catkin.
try:
    share_path = subprocess.check_output(['openrave-config', '--share-dir']).strip()
    os.environ['OPENRAVE_DATA'] = os.path.join(share_path, 'data')
except subprocess.CalledProcessError as e:
    print('error: Failed using "openrave-config" to find the default'
          ' OPENRAVE_DATA path. Loading assets may fail.',file=sys.stderr)

# Initialize OpenRAVE.
openravepy.RaveInitialize(True)
openravepy.misc.InitOpenRAVELogging()
openravepy.RaveSetDebugLevel(openravepy.DebugLevel.Fatal)


class TestTrajopt(unittest.TestCase):

    NUM_ATTEMPTS = 5
    START_CONFIG = numpy.array([0.80487864,0.42326865,-0.54016693,2.28895761,-0.34930645,-1.19702164,1.95971213])
    GOAL_CONFIG = numpy.array([2.41349473,-1.43062044,-2.69016693,2.12681216,-0.75643783,-1.52392537,1.01239878])
    TRAJECTORY_XML = """\
<trajectory>
<configuration>
<group name="joint_values BarrettWAM 0 1 2 3 4 5 6" offset="0" dof="7" interpolation="linear"/>
</configuration>
<data count="8">
0.80487864 0.42326865 -0.5401669299999999 2.28895761 -0.3493064500000005 -1.19702164 1.95971213 -0.3668195289171482 -0.1719619318499316 0.340737234496427 2.338352935126714 -1.140328948362108 -0.4582953771266394 0.1660684228974656 -0.4316746373996911 -0.1254090482184572 1.337385046499522 0.7087144047880871 -1.48802896774604 0.4679460445583862 -1.337254061021721 0.624583530772981 1.117893213994084 2.633833996473851 0.05391426760723994 -1.669197857527277 0.9002033682607622 -2.714157270255535 0.6372202491284563 1.09989072774182 2.596225701405203 0.06855704409449326 -1.662750197409222 0.8830795265988333 -2.687833191529146 1.229311742752304 0.2563870051612129 0.8340948242701349 0.7546420827296624 -1.360646074939481 0.08074456106588879 -1.454422534352764 1.821403236376152 -0.5871167174193935 -0.9280360528649325 1.440727121364831 -1.058541952469741 -0.7215904044670555 -0.2210118771763818 2.41349473 -1.43062044 -2.69016693 2.12681216 -0.75643783 -1.52392537 1.01239878 </data>
</trajectory>"""

    def setUp(self):
        self.env = openravepy.Environment()
        self.env.Load('wamtest1.env.xml')
        self.robot = self.env.GetRobot('BarrettWAM')
        self.manipulator = self.robot.GetManipulator('arm')

        with self.env:
            self.robot.SetActiveDOFs(self.manipulator.GetArmIndices())
            self.robot.SetActiveDOFValues(self.START_CONFIG)
            self.robot.SetActiveManipulator(self.manipulator)


    def tearDown(self):
        self.env.Destroy()


    def testPlanToConfiguration(self):

        with self.env:
            cspec = self.robot.GetActiveConfigurationSpecification()

            trajopt_planner = TrajoptPlanner()


            traj = trajopt_planner.PlanToConfiguration(self.robot,self.GOAL_CONFIG)

            self.assertGreaterEqual(traj.GetNumWaypoints(), 1)
            numpy.testing.assert_array_almost_equal(
            			traj.GetWaypoint(0, cspec),
            			self.START_CONFIG)
            numpy.testing.assert_array_almost_equal(
            			traj.GetWaypoint(traj.GetNumWaypoints() - 1, cspec),
            			self.GOAL_CONFIG)


    def testPlanToEndEffectorPose(self):

        with self.env:

            cspec = self.robot.GetActiveConfigurationSpecification()
            trajopt_planner = TrajoptPlanner()

            #Get pose for goal configuration
            with self.robot:
                self.robot.SetActiveDOFValues(self.GOAL_CONFIG)
                goal_pose = self.manipulator.GetEndEffectorTransform()

            print (goal_pose)
            traj = trajopt_planner.PlanToEndEffectorPose(self.robot,goal_pose)

            self.assertGreaterEqual(traj.GetNumWaypoints(), 1)
            numpy.testing.assert_array_almost_equal(
                        traj.GetWaypoint(0, cspec),
                        self.START_CONFIG)
            numpy.testing.assert_array_almost_equal(
                        traj.GetWaypoint(traj.GetNumWaypoints() - 1, cspec),
                        self.GOAL_CONFIG)


    def testPlanToIK(self):

        with self.env:

            cspec = self.robot.GetActiveConfigurationSpecification()
            trajopt_planner = TrajoptPlanner()

            #Get pose for goal configuration
            with self.robot:
                self.robot.SetActiveDOFValues(self.GOAL_CONFIG)
                goal_pose = self.manipulator.GetEndEffectorTransform()

            print (goal_pose)
            traj = trajopt_planner.PlanToIK(self.robot,goal_pose)

            self.assertGreaterEqual(traj.GetNumWaypoints(), 1)
            numpy.testing.assert_array_almost_equal(
                        traj.GetWaypoint(0, cspec),
                        self.START_CONFIG)
            numpy.testing.assert_array_almost_equal(
                        traj.GetWaypoint(traj.GetNumWaypoints() - 1, cspec),
                        self.GOAL_CONFIG)


    def testPlanToEndEffectorOffset(self):

        with self.env:

            cspec = self.robot.GetActiveConfigurationSpecification()
            trajopt_planner = TrajoptPlanner()

            directions = [[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
            distance = 0.1

            #Get start pose
            start_pose = self.manipulator.GetEndEffectorTransform()


            for direction in directions:
                
                self.robot.SetActiveDOFValues(self.START_CONFIG)

                dir_np = numpy.array(direction)
                dist = numpy.transpose(distance*dir_np/numpy.linalg.norm(dir_np))

                goal_pose = start_pose
                goal_pose[0:3,3] = start_pose[0:3,3] + dist 

                traj = trajopt_planner.PlanToEndEffectorOffset(self.robot,direction,distance)

                self.robot.SetActiveDOFValues(traj.GetWaypoint(traj.GetNumWaypoints()-1,cspec))
                goal_pose_returned = self.manipulator.GetEndEffectorTransform()

                self.assertGreaterEqual(traj.GetNumWaypoints(), 1)
                numpy.testing.assert_array_almost_equal(goal_pose,goal_pose_returned)

if __name__ == '__main__':
    unittest.main()