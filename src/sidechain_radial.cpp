#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include "affine.h"
#include <cmath>
#include <vector>
#include "interaction_graph.h"
#include "spline.h"
#include "state_logger.h"

using namespace std;
using namespace h5;

namespace {
template <bool is_symmetric>
struct RadialHelper {
    // spline-based distance interaction
    // n_knot is the number of basis splines (including those required to get zero
    //   derivative in the clamped spline)
    // spline is constant over [0,dx] to avoid funniness at origin

    // Please obey these 4 conditions:
    // p[0] = 1./dx, that is the inverse of the knot spacing
    // should have p[1] == p[3] for origin clamp (p[0] is inv_dx)
    // should have p[-3] == p[-1] (negative indices from the end, Python-style) for terminal clamping
    // should have (1./6.)*p[-3] + (2./3.)*p[-2] + (1./6.)*p[-1] == 0. for continuity at cutoff

    constexpr static bool  symmetric = is_symmetric;
    constexpr static int   n_knot=16, n_param=1+n_knot, n_dim1=3, n_dim2=3, simd_width=1;

    static float cutoff(const float* p) {
        const float inv_dx = p[0];
        return (n_knot-2-1e-6)/inv_dx;  // 1e-6 just insulates us from round-off error
    }

    static bool is_compatible(const float* p1, const float* p2) {
        if(symmetric) for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
        return true;
    }

    static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2, const Int4& l_start) {
        auto sequence_exclude = Int4(2);
        return (sequence_exclude < id1-id2) | (sequence_exclude < id2-id1);
    }

    static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4], 
            const Vec<n_dim1,Float4> &x1, const Vec<n_dim2,Float4> &x2) {
        alignas(16) const float inv_dx_data[4] = {p[0][0], p[1][0], p[2][0], p[3][0]};

        auto inv_dx     = Float4(inv_dx_data, Alignment::aligned);
        auto disp       = x1-x2;
        auto dist2      = mag2(disp);
        auto inv_dist   = rsqrt(dist2+Float4(1e-7f));  // 1e-7 is divergence protection
        auto dist_coord = dist2*(inv_dist*inv_dx);

        const float* pp[4] = {p[0]+1, p[1]+1, p[2]+1, p[3]+1};
        auto en = clamped_deBoor_value_and_deriv(pp, dist_coord, n_knot);
        d1 = disp*(inv_dist*inv_dx*en.y());
        d2 = -d1;
        return en.x();
    }

    static void param_deriv(Vec<n_param> &d_param, const float* p,
            const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
        d_param = make_zero<n_param>();
        float inv_dx = p[0];
        auto dist_coord = inv_dx*mag(x1-x2); // to convert to spline coords of interger grid of knots
        auto dV_dinv_dx = clamped_deBoor_value_and_deriv(p+1, dist_coord, n_knot).y()*mag(x1-x2);
        d_param[0] = dV_dinv_dx;
 
        int starting_bin;
        float result[4];
        clamped_deBoor_coeff_deriv(&starting_bin, result, dist_coord, n_knot);
        for(int i: range(4)) d_param[1+starting_bin+i] = result[i];
   }
};



struct SidechainRadialPairs : public PotentialNode
{

    InteractionGraph<RadialHelper<true>> igraph;

    SidechainRadialPairs(hid_t grp, CoordNode& bb_point_):
        PotentialNode(),
        igraph(grp, &bb_point_)
    {};

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("radial_pairs"));

        igraph.compute_edges();
        for(int ne=0; ne<igraph.n_edge; ++ne) igraph.edge_sensitivity[ne] = 1.f;
        igraph.propagate_derivatives();

        if(mode==PotentialAndDerivMode) {
            potential = 0.f;
            for(int ne=0; ne<igraph.n_edge; ++ne) 
                potential += igraph.edge_value[ne];
        }
    }
};


struct HBondSidechainRadialPairs : public PotentialNode
{

    InteractionGraph<RadialHelper<false>> igraph;

    HBondSidechainRadialPairs(hid_t grp, CoordNode& hb_point_, CoordNode& bb_point_):
        PotentialNode(),
        igraph(grp, &hb_point_, &bb_point_)
    {};

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("hbond_sc_radial_pairs"));

        igraph.compute_edges();
        for(int ne=0; ne<igraph.n_edge; ++ne) igraph.edge_sensitivity[ne] = 1.f;
        igraph.propagate_derivatives();

        if(mode==PotentialAndDerivMode) {
            potential = 0.f;
            for(int ne=0; ne<igraph.n_edge; ++ne) 
                potential += igraph.edge_value[ne];
        }
    }

    virtual std::vector<float> get_param() const override {return igraph.get_param();}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {return igraph.get_param_deriv();}
#endif
    virtual void set_param(const std::vector<float>& new_param) override {igraph.set_param(new_param);}
};


struct ContactEnergy : public PotentialNode
{
    struct Param {
        index_t   loc[2];
        float     energy;
        float     dist;
        float     scale;  // 1.f/width
        float     cutoff;
    };

    int n_contact;
    CoordNode& bead_pos;
    vector<Param> params;
    float cutoff;

    ContactEnergy(hid_t grp, CoordNode& bead_pos_):
        PotentialNode(),
        n_contact(get_dset_size(2, grp, "id")[0]),
        bead_pos(bead_pos_), 
        params(n_contact)
    {
        check_size(grp, "id",       n_contact, 2);
        check_size(grp, "energy",   n_contact);
        check_size(grp, "distance", n_contact);
        check_size(grp, "width",    n_contact);

        traverse_dset<2,int  >(grp, "id",       [&](size_t nc, size_t i, int x){params[nc].loc[i] = x;});
        traverse_dset<1,float>(grp, "distance", [&](size_t nc, float x){params[nc].dist = x;});
        traverse_dset<1,float>(grp, "energy",   [&](size_t nc, float x){params[nc].energy = x;});
        traverse_dset<1,float>(grp, "width",    [&](size_t nc, float x){params[nc].scale = 1.f/x;});
        for(auto &p: params) p.cutoff = p.dist + 1.f/p.scale;

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("contact_energy", {bead_pos.n_elem}, 
                    [&](float* buffer) {
                       fill_n(buffer, bead_pos.n_elem, 0.f);
                       VecArray pos  = bead_pos.output;

                       for(const auto &p: params) {
                           auto dist = mag(load_vec<3>(pos, p.loc[0]) - load_vec<3>(pos, p.loc[1]));
                           float en = p.energy * compact_sigmoid(dist-p.dist, p.scale)[0];
                           buffer[p.loc[0]] += 0.5f*en;
                           buffer[p.loc[1]] += 0.5f*en;
                       }});
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("contact_energy"));
        VecArray pos  = bead_pos.output;
        VecArray sens = bead_pos.sens;
        potential = 0.f;

        for(int nc=0; nc<n_contact; ++nc) {
            const auto& p = params[nc];
            auto disp = load_vec<3>(pos, p.loc[0]) - load_vec<3>(pos, p.loc[1]);
            auto dist = mag(disp);
            if(dist>=p.cutoff) continue;

            Vec<2> contact = compact_sigmoid(dist-p.dist, p.scale);
            potential += p.energy*contact.x();
            auto deriv = (p.energy*contact.y()*rcp(dist)) * disp;
            update_vec(sens, p.loc[0],  deriv);
            update_vec(sens, p.loc[1], -deriv);
        }
    }
};

struct CooperationContacts : public PotentialNode
{
    struct Param {
        index_t   loc[2];
    };

    float energy;
    int n_contact;
    CoordNode& bead_pos;
    vector<Param> params;
    float p_energy;
    float p_dist;
    float p_scale;  // 1.f/width
    float p_cutoff;
    vector<float> dists;
    vector<Vec<2> > contacts;

    CooperationContacts(hid_t grp, CoordNode& bead_pos_):
        PotentialNode(),
        n_contact(get_dset_size(2, grp, "id")[0]),
        bead_pos(bead_pos_), 
        params(n_contact),
        p_energy(read_attribute<float>(grp, ".", "energy")),
        p_dist(read_attribute<float>(grp, ".", "dist")),
        p_scale(1.f/read_attribute<float>(grp, ".", "width")),
        p_cutoff(p_dist + 1.f/p_scale),
        dists(n_contact),
        contacts(n_contact)
    {
        check_size(grp, "id",       n_contact, 2);

        traverse_dset<2,int  >(grp, "id",       [&](size_t nc, size_t i, int x){params[nc].loc[i] = x;});

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("cooperation_contacts", {bead_pos.n_elem}, 
                    [&](float* buffer) {
                       fill_n(buffer, bead_pos.n_elem, 0.f);
                       VecArray pos  = bead_pos.output;

                       float en = 1.f;
                       for(const auto &p: params) {
                           auto dist = mag(load_vec<3>(pos, p.loc[0]) - load_vec<3>(pos, p.loc[1]));
                           en *= compact_sigmoid(dist-p_dist, p_scale)[0];
                       }
                       en = p_energy * en;

                       for(const auto &p: params) {
                           buffer[p.loc[0]] += 0.5f*en;
                           buffer[p.loc[1]] += 0.5f*en;
                       }
                  }
             );
         }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("cooperation_contacts"));

        VecArray pos  = bead_pos.output;
        VecArray sens = bead_pos.sens;

        float product_contact = 1.f;

        // calc potential energy
        for(int nc=0; nc<n_contact; ++nc) {
            const auto& p = params[nc];
            auto disp = load_vec<3>(pos, p.loc[0]) - load_vec<3>(pos, p.loc[1]);
            auto dist = mag(disp);
            dists[nc] = dist;
            Vec<2> contact = compact_sigmoid(dist-p_dist, p_scale);
            contacts[nc] = contact;
            product_contact *= contact.x();
        }
        potential = p_energy * product_contact;

        // calc force
        if( product_contact > 0.f and product_contact<1.f) {
            for(int nc=0; nc<n_contact; ++nc) {
                const auto& p = params[nc];
                auto dist = dists[nc];

                auto disp = load_vec<3>(pos, p.loc[0]) - load_vec<3>(pos, p.loc[1]);
                auto contact = contacts[nc];
                auto deriv = potential*contact.y()*rcp(contact.x())*rcp(dist) * disp;

                update_vec(sens, p.loc[0],  deriv);
                update_vec(sens, p.loc[1], -deriv);
            }
        }
    }
};

}

static RegisterNodeType<ContactEnergy,1>             contact_node("contact");
static RegisterNodeType<CooperationContacts,1>       cooperation_contacts_node("cooperation_contacts");
static RegisterNodeType<SidechainRadialPairs,1>      radial_node ("radial");
static RegisterNodeType<HBondSidechainRadialPairs,2> hbond_sc_radial_node ("hbond_sc_radial");
