#ifndef g2o_MSC_H
#define g2o_MSC_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"

namespace g2o {
    inline double ComputeAngle2D(const Vector3d &l_a_hom, const Vector2d &l_b) {
        // Compute the angle between two vectors
        Eigen::Vector2d l_a(l_a_hom[0] / l_a_hom[2], l_a_hom[1] / l_a_hom[2]);

        double dot_product = l_b.dot(l_a);

        // Find magnitude of line AB and BC
        double magn_a = l_b.norm();
        double magn_b = l_a.norm();

        // Find the cosine of the angle formed
        return abs(dot_product / (magn_a * magn_b));
    }

    inline double ComputeAngle3D(const Vector3d &l_a, const Vector3d &l_b) {
        double dot_product = l_b.dot(l_a);

        // Find magnitude of lines
        double magn_a = l_b.norm();
        double magn_b = l_a.norm();

        // Find the cosine of the angle formed
        return abs(dot_product / (magn_a * magn_b));
    }

    inline Vector2d project2d(const Vector3d &pt) {
        Vector2d res;
        res(0) = pt(0) / pt(2);
        res(1) = pt(1) / pt(2);
        return res;
    }

    inline Vector2d
    cam_project(const Vector3d &pt, const double &fx, const double &fy, const double &cx, const double &cy) {
        ///归一化
        Vector2d proj_pt = project2d(pt);
        Vector2d result;
        ///像素坐标
        result[0] = proj_pt[0] * fx + cx;
        result[1] = proj_pt[1] * fy + cy;
        return result;
    }

    inline Vector3d vp_project(const Vector3d &s_pt) {
        Vector3d result;
        result[0] = s_pt[0];
        result[1] = s_pt[1];
        result[2] = s_pt[2];
        return result;
    }

    class ParVPSingleFrame
            : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ParVPSingleFrame() {}

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            Vector3d line1 = end_pt->estimate() - st_pt->estimate();
            Vector3d line2 = _measurement;
            if(line1.dot(line2) < 0.0)line1 = st_pt->estimate() - end_pt->estimate();
            ///同向
            Vector3d vp1(fx *line1[0]+cx * line1[2], fy * line1[1] + cy * line1[2], line1[2]);
            Vector3d vp2(fx *line2[0]+cx * line2[2], fy * line2[1] + cy * line2[2], line2[2]);
            if(vp1[2]==0.0 || vp2[2]==0.0)
            {
                cout<<"exist 0"<<endl;
                error = 1.0 * 0x3fffffff;
                return ;
            }
            vp1[0]/=vp1[2],vp1[1]/=vp1[2];
            vp2[0]/=vp2[2],vp2[1]/=vp2[2];
            vp1 = vp1 / sqrt(vp1[0]*vp1[0] + vp1[1]*vp1[1] );
            vp2 = vp2 / sqrt(vp2[0]*vp2[0] + vp2[1]*vp2[1] );
            _error(0) = sqrt(pow(vp1[0]-vp2[0],2) + pow(vp1[1]-vp2[1],2));
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
        double fx, fy, cx, cy,error=0.0;
    };
    class ParEptsNVector3DSingleFrame
            : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ParEptsNVector3DSingleFrame() {}

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            Vector3d line_eq = end_pt->estimate() - st_pt->estimate();
            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = 1 - error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class PerpEptsNVector3DSingleFrame
            : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PerpEptsNVector3DSingleFrame() {}

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            // Extract 3D line eq. from two endpoints
            Vector3d line_eq = end_pt->estimate() - st_pt->estimate();

            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class ParEptsNVector2DSingleFrame
            : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ParEptsNVector2DSingleFrame() {}

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            // Project 3D endpoints to image coordinates
            Vector2d proj_st_pt = cam_project(st_pt->estimate(), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(end_pt->estimate(), fx, fy, cx, cy);

            // Extract 2D line equation
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = 1 - angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class PerpEptsNVector2DSingleFrame
            : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PerpEptsNVector2DSingleFrame() {}

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            // Project 3D endpoints to image coordinates
            Vector2d proj_st_pt = cam_project(st_pt->estimate(), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(end_pt->estimate(), fx, fy, cx, cy);
            // Extract 2D line equation
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class ParEptsNVector3DMultiFrame : public BaseMultiEdge<3, Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ParEptsNVector3DMultiFrame() {
            resize(3);
        }

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates
            Vector3d st_pt_rot = poseVertex->estimate().map(st_pt->estimate());
            Vector3d end_pt_rot = poseVertex->estimate().map(end_pt->estimate());
            // Extract line eq.
            Vector3d line_eq = end_pt_rot - st_pt_rot;

            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = 1 - error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class PerpEptsNVector3DMultiFrame : public BaseMultiEdge<3, Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PerpEptsNVector3DMultiFrame() {
            resize(3);
        }

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates
            Vector3d st_pt_rot = poseVertex->estimate().map(st_pt->estimate());
            Vector3d end_pt_rot = poseVertex->estimate().map(end_pt->estimate());

            // Extract line eq.
            Vector3d line_eq = end_pt_rot - st_pt_rot;

            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class ParEptsNVector2DMultiFrame : public BaseMultiEdge<3, Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ParEptsNVector2DMultiFrame() {
            resize(3);
        }

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *pose_vertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates and project them to image coordinates
            Vector2d proj_st_pt = cam_project(pose_vertex->estimate().map(st_pt->estimate()), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(pose_vertex->estimate().map(end_pt->estimate()), fx, fy, cx, cy);

            // Extract line eq.
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            // Note that the _measurement is in homogeneous coordinates
            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = 1 - angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class PerpEptsNVector2DMultiFrame : public BaseMultiEdge<3, Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PerpEptsNVector2DMultiFrame() {
            resize(3);
        }

        virtual void computeError() {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *pose_vertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates and project them to image coordinates
            Vector2d proj_st_pt = cam_project(pose_vertex->estimate().map(st_pt->estimate()), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(pose_vertex->estimate().map(end_pt->estimate()), fx, fy, cx, cy);

            // Extract line eq.
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            // Note that the _measurement is in homogeneous coordinates
            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class DistPt2Line2DMultiFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        DistPt2Line2DMultiFrame() {}

        virtual void computeError() {
            const VertexSE3Expmap *v1 = static_cast<VertexSE3Expmap *>(_vertices[1]);
            const VertexSBAPointXYZ *v2 = static_cast<VertexSBAPointXYZ *>(_vertices[0]);

            ///这里的obs貌似没用
            Vector3d obs = _measurement;
            Vector2d proj_pt = cam_project(v1->estimate().map(v2->estimate()), fx, fy, cx, cy);

            // Point to line distance in image coordinates
            _error(0) = obs(0) * proj_pt(0) + obs(1) * proj_pt(1) + obs(2);
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class DistPt2Line2DMultiFrameOnlyPose : public BaseUnaryEdge<3, Vector3d, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        DistPt2Line2DMultiFrameOnlyPose() {}

        virtual void computeError() {
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Vector3d obs_hom_coord = _measurement;
            // Transform endpoint from world to frame, and project it in img coord.
            Vector2d pt_proj = cam_project(v1->estimate().map(Xw), fx, fy, cx, cy);

            // Dist Point to line
            ///这是地图线的起点投影到像素平面与线特征的一个距离
            _error(0) = obs_hom_coord(0) * pt_proj(0) + obs_hom_coord(1) * pt_proj(1) + obs_hom_coord(2);
            _error(1) = 0;
            _error(2) = 0;
            //cout<<"点线距离 = "<<_error(0)<<endl;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        Vector3d Xw; // Non optimizable line endpoint in world coord.
        double fx, fy, cx, cy;
    };

    class Par2Vectors3DMultiFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Par2Vectors3DMultiFrame() {}

        virtual void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[1]);
            const VertexSBAPointXYZ *manhAxisVertex = static_cast<VertexSBAPointXYZ *>(_vertices[0]);

            // Rotate the associated Manh. axis
            Eigen::Vector3d manh_axis_vertex = manhAxisVertex->estimate();
            const g2o::SE3Quat tf = poseVertex->estimate();
            const Eigen::Quaterniond w2n_quat = tf.rotation();
            Eigen::Vector3d fr_coord_line_manh = w2n_quat * manh_axis_vertex;

            // Manh. axis and 3D line eq. frame angle difference
            _error[0] = 1 - ComputeAngle3D(fr_coord_line_manh, _measurement);
            _error[1] = 0.0;
            _error[2] = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class Perp2Vectors3DMultiFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Perp2Vectors3DMultiFrame() {}

        virtual void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[1]);
            const VertexSBAPointXYZ *manhAxisVertex = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
            // Rotate the associated Manh. axis
            Eigen::Vector3d manh_axis_vertex = manhAxisVertex->estimate();
            const g2o::SE3Quat tf = poseVertex->estimate();
            const Eigen::Quaterniond w2n_quat = tf.rotation();
            Eigen::Vector3d fr_coord_line_manh = w2n_quat * manh_axis_vertex;

            // Manh. axis and 3D line eq. frame angle difference
            _error[0] = ComputeAngle3D(fr_coord_line_manh, _measurement);
            _error[1] = 0.0;
            _error[2] = 0.0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    ///vanish point g2o
    class DistVp2VpOnlyPose : public BaseUnaryEdge<3, Vector3d, g2o::VertexSE3Expmap>
            {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DistVp2VpOnlyPose() {}

        virtual void computeError() {
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Vector3d obs_hom_coord = _measurement;
            ///将地图线从世界坐标系转到相机坐标系 然后利用端点计算消影点
            ///通过当前帧的位姿 将地图线的两个端点转换到相机坐标系下
            Vector3d vp_proj1 = vp_project(v1->estimate().map(st));
            Vector3d vp_proj2 = vp_project(v1->estimate().map(et));
            ///利用相机坐标系下的两个端点计算出线段的方向向量
            Vector3d dc(vp_proj2[0] - vp_proj1[0], vp_proj2[1] - vp_proj1[1], vp_proj2[2] - vp_proj1[2]);
            ///判断两个方向向量是否同向
            if (obs_hom_coord.dot(dc) / (sqrt(dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]) *
                                         sqrt(obs_hom_coord[0] * obs_hom_coord[0] +
                                              obs_hom_coord[1] * obs_hom_coord[1] +
                                              obs_hom_coord[2] * obs_hom_coord[2])) < 0.0) {
                Vector3d dc2(vp_proj1[0] - vp_proj2[0], vp_proj1[1] - vp_proj2[1], vp_proj1[2] - vp_proj2[2]);
                dc = dc2;
            }
            ///这里已经完成了两个方向向量的匹配  接下来利用方向向量计算消影点
            Vector3d vp1(fx *_measurement[0]
            +cx * _measurement[2], fy * _measurement[1] + cy * _measurement[2], _measurement[2]);
            Vector3d vp2(fx *dc[0]
            +cx * dc[2], fy * dc[1] + cy * dc[2], dc[2]);

            if(vp1[2]==0.0 || vp2[2]==0.0)
            {
                //cout<<"exist 0"<<endl;
                error = 1.0 * 0x3fffffff;
                return ;
            }
            vp1[0]/=vp1[2],vp1[1]/=vp1[2];
            vp2[0]/=vp2[2],vp2[1]/=vp2[2];
            vp1 = vp1 / sqrt(vp1[0]*vp1[0] + vp1[1]*vp1[1] );
            vp2 = vp2 / sqrt(vp2[0]*vp2[0] + vp2[1]*vp2[1] );
            //cout<<"vp1 = "<<vp1<<endl;
            //cout<<"vp2 = "<<vp2<<endl;
            ///dist vp to vp
            _error(0) = sqrt(pow(vp1[0] - vp2[0], 2) + pow(vp1[1] - vp2[1], 2) );
            _error(1) = 0;
            _error(2) = 0;
            error = _error(0);
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }
        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }


    double fx, fy, cx, cy;
    Vector3d st, et; // non optimizable line endpoint in world coord
    double error=0.0;
    };

    ///point on line g2o
    class DistP2LOnlyPose : public BaseUnaryEdge<3, Vector3d, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DistP2LOnlyPose() {}

        virtual void computeError()
        {
            ///Tcw
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            ///将像素坐标下的归一化线函数作为测量值
            Vector3d line(_measurement);
            ///将地图点重投影到像素平面
            Vector2d obs = cam_project(v1->estimate().map(Xw),fx,fy,cx,cy);
            /// 计算重投影到像素平面的地图点到line的距离  作为error
            _error(0) = (obs[0]*line[0] + obs[1]*line[1] + line[2]) / std::sqrt(line[0]*line[0] + line[1]*line[1]);
            _error(1) = 0;
            _error(2) = 0;

            //cout<<"error = "<<_error(0)<<endl;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }
        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
        double fx, fy, cx, cy;
        Vector3d Xw; // pixel corrdinate line
    };
};





#endif // g2o_MSC_H
