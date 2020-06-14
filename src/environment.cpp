#include "deriv_engine.h"
#include "timing.h"
#include "state_logger.h"
#include "spline.h"
#include "interaction_graph.h"
#include <algorithm>
#include <iostream>

#define NUM_AMINO_ACID_TYPE 20

using namespace std;
using namespace h5;

namespace {
    struct EnvironmentCoverageInteraction {
        // parameters are r0,r_sharpness, dot0,dot_sharpness

        constexpr static bool  symmetric = false;
        constexpr static int   n_param=4, n_dim1=6, n_dim2=4, simd_width=1;

        static float cutoff(const float* p) {
            return p[0] + compact_sigmoid_cutoff(p[1]);
        }

        static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2, const Int4& l_start) {
            auto sequence_exclude = Int4(2);  // exclude i,i, i,i+1, and i,i+2
            return (sequence_exclude < id1-id2) | (sequence_exclude < id2-id1);
        }

        static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4],
                const Vec<n_dim1,Float4> &cb_pos, const Vec<n_dim2,Float4> &sc_pos) {
            Float4 one(1.f);
            auto displace = extract<0,3>(sc_pos)-extract<0,3>(cb_pos);
            auto rvec1 = extract<3,6>(cb_pos);
            auto prob  = sc_pos[3];

            auto dist2 = mag2(displace);
            auto inv_dist = rsqrt(dist2);
            auto dist = dist2*inv_dist;
            auto displace_unitvec = inv_dist*displace;

            // read parameters then transpose
            Float4 r0(p[0]);
            Float4 r_sharpness(p[1]);
            Float4 dot0(p[2]);
            Float4 dot_sharpness(p[3]);
            transpose4(r0,r_sharpness,dot0,dot_sharpness);

            auto dp = dot(displace_unitvec,rvec1);
            auto radial_sig  = compact_sigmoid(dist-r0, r_sharpness);
            auto angular_sig = compact_sigmoid(dot0-dp, dot_sharpness);

            // now we compute derivatives (minus sign is from the derivative of angular_sig)
            auto d_displace = prob*(radial_sig.y()*angular_sig.x() * displace_unitvec -
                                    radial_sig.x()*angular_sig.y()* inv_dist*(rvec1 - dp*displace_unitvec));

            store<3,6>(d1, -prob*radial_sig.x()*angular_sig.y()*displace_unitvec);
            store<0,3>(d1, -d_displace);
            store<0,3>(d2,  d_displace);
            auto score = radial_sig.x() * angular_sig.x();
            d2[3] = score;
            return prob * score;
        }

        static void param_deriv(Vec<n_param> &d_param, const float* p,
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            d_param = make_zero<n_param>();   // not implemented currently
        }

        static bool is_compatible(const float* p1, const float* p2) {return true;};
    };

struct EnvironmentCoverage : public CoordNode {

    int n_aa;
    int n_res;
    std::vector<int> aa_types;
    InteractionGraph<EnvironmentCoverageInteraction> igraph;

    EnvironmentCoverage(hid_t grp, CoordNode& cb_pos_, CoordNode& weighted_sidechains_):
        CoordNode(get_dset_size(1,grp,"index1")[0]*NUM_AMINO_ACID_TYPE, 1),
        n_aa(NUM_AMINO_ACID_TYPE), 
        n_res(get_dset_size(1,grp,"index1")[0]),
        aa_types(n_res),
        igraph(grp, &cb_pos_, &weighted_sidechains_)
    {
        check_size(grp, "aa_types", n_res);
        traverse_dset<1,int>  (grp,"aa_types",[&](size_t ne, int x){aa_types[ne]=x;});

        if(logging(LOG_EXTENSIVE)) {
            default_logger->add_logger<float>("environment_coverage", {n_elem}, [&](float* buffer) {
                    for(int ne: range(n_elem))
                            buffer[ne] = output(0,ne);});
        }

        for(int i=0;i<n_aa*n_res; ++i)
            output(0, i) = 0.0;
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("environment_coverage"));

        igraph.compute_edges();

        fill(output, 0.f);

        // accumulate for each cb
        for(int ne=0; ne<igraph.n_edge; ++ne) {
            int indices1 = igraph.edge_indices1[ne];
            int id2      = igraph.edge_id2[ne];
            int type2    = aa_types[id2];
            output(0, indices1*n_aa+type2) += igraph.edge_value[ne];
        }
    }

    virtual void propagate_deriv() override {
        Timer timer(string("d_environment_coverage"));

        for(int ne: range(igraph.n_edge)) {
            int indices1 = igraph.edge_indices1[ne];
            int id2      = igraph.edge_id2[ne];
            int type2    = aa_types[id2];
            igraph.edge_sensitivity[ne] = sens(0, indices1*n_aa+type2);
        }
        igraph.propagate_derivatives();
    }

    virtual std::vector<float> get_param() const override {return igraph.get_param();}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {return igraph.get_param_deriv();}
#endif
    virtual void set_param(const std::vector<float>& new_param) override {igraph.set_param(new_param);}
};
static RegisterNodeType<EnvironmentCoverage,2> environment_coverage_node("environment_coverage");

struct InterEnvironmentCoverage : public CoordNode {

    int n_aa;
    int n_res;
    int l_start;
    std::vector<int> aa_types;
    InteractionGraph<EnvironmentCoverageInteraction> igraph;

    InterEnvironmentCoverage(hid_t grp, CoordNode& cb_pos_, CoordNode& weighted_sidechains_):
        CoordNode(get_dset_size(1,grp,"index1")[0]*NUM_AMINO_ACID_TYPE, 2), // initializes CoordNode output w/2 rows, n_res*n_aa columns
        n_aa(NUM_AMINO_ACID_TYPE), 
        n_res(get_dset_size(1,grp,"index1")[0]),
        l_start(h5::read_attribute<int32_t>(grp, ".", "l_start", 0)),
        aa_types(n_res),
        igraph(grp, &cb_pos_, &weighted_sidechains_)
    {
        check_size(grp, "aa_types", n_res);
        traverse_dset<1,int>  (grp,"aa_types",[&](size_t ne, int x){aa_types[ne]=x;});

        if(logging(LOG_EXTENSIVE)) {
            default_logger->add_logger<float>("inter_environment_coverage", {n_elem}, [&](float* buffer) {
                    for(int ne: range(n_elem))
                            buffer[ne] = output(0,ne);});
        }

        for(int i=0;i<n_aa*n_res; ++i)
            output(0, i) = 0.0;
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("inter_environment_coverage"));

        igraph.compute_edges();

        fill(output, 0.f);

        // accumulate for each cb
        for(int ne=0; ne<igraph.n_edge; ++ne) {
            int indices1 = igraph.edge_indices1[ne];
            int id2      = igraph.edge_id2[ne];
            int type2    = aa_types[id2];

            if (indices1 < l_start && id2 >= l_start)
                output(1, indices1) += 1; // second row of output holds num inter beads

            output(0, indices1*n_aa+type2) += igraph.edge_value[ne];
        }

        // for(int i=0;i<n_res; ++i)
        //     if (output(1, indices1) > 0) printf("%d: %d\n", i, output(1, indices1));
    }

    virtual void propagate_deriv() override {
        Timer timer(string("d_inter_environment_coverage"));

        for(int ne: range(igraph.n_edge)) {
            int indices1 = igraph.edge_indices1[ne];
            int id2      = igraph.edge_id2[ne];
            int type2    = aa_types[id2];
            igraph.edge_sensitivity[ne] = sens(0, indices1*n_aa+type2);
        }
        igraph.propagate_derivatives();
    }

    virtual std::vector<float> get_param() const override {return igraph.get_param();}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {return igraph.get_param_deriv();}
#endif
    virtual void set_param(const std::vector<float>& new_param) override {igraph.set_param(new_param);}
};
static RegisterNodeType<InterEnvironmentCoverage,2> inter_environment_coverage_node("inter_environment_coverage");

struct WeightedPos : public CoordNode {
    CoordNode &pos;
    CoordNode &energy;

    struct Param {
        index_t index_pos;
        index_t index_weight;
    };
    vector<Param> params;

    WeightedPos(hid_t grp, CoordNode& pos_, CoordNode& energy_):
        CoordNode(get_dset_size(1,grp,"index_pos")[0], 4),
        pos(pos_),
        energy(energy_),
        params(n_elem)
    {
        traverse_dset<1,int>(grp,"index_pos"   ,[&](size_t ne, int x){params[ne].index_pos   =x;});
        traverse_dset<1,int>(grp,"index_weight",[&](size_t ne, int x){params[ne].index_weight=x;});
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("weighted_pos");

        for(int ne=0; ne<n_elem; ++ne) {
            auto p = params[ne];
            output(0,ne) = pos.output(0,p.index_pos);
            output(1,ne) = pos.output(1,p.index_pos);
            output(2,ne) = pos.output(2,p.index_pos);
            output(3,ne) = expf(-energy.output(0,p.index_weight));
        }
    }

    virtual void propagate_deriv() override {
        Timer timer("d_weighted_pos");

        for(int ne=0; ne<n_elem; ++ne) {
            auto p = params[ne];
            pos.sens(0,p.index_pos) += sens(0,ne);
            pos.sens(1,p.index_pos) += sens(1,ne);
            pos.sens(2,p.index_pos) += sens(2,ne);
            energy.sens(0,p.index_weight) -= output(3,ne)*sens(3,ne); // exponential derivative
        }
    }
};
static RegisterNodeType<WeightedPos,2> weighted_pos_node("weighted_pos");

struct UniformTransform : public CoordNode {
    CoordNode& input;
    int n_coeff;
    float spline_offset;
    float spline_inv_dx;
    unique_ptr<float[]> bspline_coeff;
    unique_ptr<float[]> jac;


    UniformTransform(hid_t grp, CoordNode& input_):
        CoordNode(input_.n_elem, 1),
        input(input_),
        n_coeff(get_dset_size(1,grp,"bspline_coeff")[0]),
        spline_offset(read_attribute<float>(grp,"bspline_coeff","spline_offset")),
        spline_inv_dx(read_attribute<float>(grp,"bspline_coeff","spline_inv_dx")),
        bspline_coeff(new_aligned<float>(n_coeff,4)),
        jac          (new_aligned<float>(input.n_elem,4))
    {
        check_elem_width(input,1); // this restriction could be lifted
        fill_n(bspline_coeff.get(), round_up(n_coeff,4), 0.f);
        traverse_dset<1,float>(grp,"bspline_coeff",[&](size_t ne, float x){bspline_coeff[ne]=x;});
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("uniform_transform");
        for(int ne=0; ne<n_elem; ++ne) {
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(bspline_coeff.get(), coord, n_coeff);
            output(0,ne) = v[0];
            jac[ne]      = v[1]*spline_inv_dx;
        }
    }

    virtual void propagate_deriv() override {
        Timer timer("d_uniform_transform");
        for(int ne=0; ne<n_elem; ++ne)
            input.sens(0,ne) += jac[ne]*sens(0,ne);
    }

    virtual std::vector<float> get_param() const override{
        vector<float> ret(2+n_coeff);
        ret[0] = spline_offset;
        ret[1] = spline_inv_dx;
        copy_n(bspline_coeff.get(), n_coeff, &ret[2]);
        return ret;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {
        vector<float> ret(2+n_coeff, 0.f);
        int starting_bin;
        float d[4];
        for(int ne=0; ne<n_elem; ++ne) {
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(bspline_coeff.get(), coord, n_coeff);
            clamped_deBoor_coeff_deriv(&starting_bin, d, coord, n_coeff);

            ret[0] += v[1];                                    // derivative of offset
            ret[1] += v[1]*(input.output(0,ne)-spline_offset); // derivative of inv_dx
            for(int i: range(4)) ret[2+starting_bin+i] += d[i];
        }
        return ret;
    }
#endif

    virtual void set_param(const std::vector<float>& new_param) override {
        if(new_param.size() < size_t(2+4)) throw string("too small of size for spline");
        if(int(new_param.size())-2 != n_coeff) {
            n_coeff = int(new_param.size())-2;
            bspline_coeff = new_aligned<float>(n_coeff,4);
            fill_n(bspline_coeff.get(), round_up(n_coeff,4), 0.f);
        }
        spline_offset = new_param[0];
        spline_inv_dx = new_param[1];
        copy_n(begin(new_param)+2, n_coeff, bspline_coeff.get());
    }
};
static RegisterNodeType<UniformTransform,1> uniform_transform_node("uniform_transform");

struct LinearCoupling : public PotentialNode {
    CoordNode& input;
    vector<float> couplings;      // length n_restype
    vector<int>   coupling_types; // length input n_elem
    CoordNode* inactivation;  // 0 to 1 variable to inactivate energy
    int inactivation_dim;

    LinearCoupling(hid_t grp, CoordNode& input_, CoordNode& inactivation_):
        LinearCoupling(grp, &input_, &inactivation_) {}

    LinearCoupling(hid_t grp, CoordNode& input_):
        LinearCoupling(grp, &input_, nullptr) {}

    LinearCoupling(hid_t grp, CoordNode* input_, CoordNode* inactivation_):
        PotentialNode(),
        input(*input_),
        couplings(get_dset_size(1,grp,"couplings")[0]),
        coupling_types(input.n_elem),
        inactivation(inactivation_),
        inactivation_dim(inactivation ? read_attribute<int>(grp, ".", "inactivation_dim") : 0)
    {
        check_elem_width(input, 1);  // could be generalized

        if(inactivation) {
            if(input.n_elem != inactivation->n_elem)
                throw string("Inactivation size must match input size");
            check_elem_width_lower_bound(*inactivation, inactivation_dim+1);
        }

        check_size(grp, "coupling_types", input.n_elem);
        traverse_dset<1,float>(grp,"couplings",[&](size_t nt, float x){couplings[nt]=x;});
        traverse_dset<1,int>(grp,"coupling_types",[&](size_t ne, int x){coupling_types[ne]=x;});
        for(int i: coupling_types) if(i<0 || i>=int(couplings.size())) throw string("invalid coupling type");

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>(
                    (inactivation ? "linear_coupling_with_inactivation" : "linear_coupling_uniform"),
                    {input.n_elem}, [&](float* buffer) {
                    for(int ne: range(input.n_elem)) {
                        float c = couplings[coupling_types[ne]];
                        buffer[ne] = c*input.output(0,ne);
                    }});
        }
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("linear_coupling");
        int n_elem = input.n_elem;
        float pot = 0.f;
        for(int ne=0; ne<n_elem; ++ne) {
            float c = couplings[coupling_types[ne]];
            float act = inactivation ? sqr(1.f-inactivation->output(inactivation_dim,ne)) : 1.f;
            float val = input.output(0,ne);
            pot += c * val * act;
            input.sens(0,ne) += c*act;
            if(inactivation) inactivation->sens(inactivation_dim,ne) -= c*val;
        }
        potential = pot;
    }

    virtual std::vector<float> get_param() const override {
        return couplings;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {
        vector<float> deriv(couplings.size(), 0.f);

        int n_elem = input.n_elem;
        for(int ne=0; ne<n_elem; ++ne) {
            float act = inactivation ? 1.f - inactivation->output(inactivation_dim,ne) : 1.f;
            deriv[coupling_types[ne]] += input.output(0,ne) * act;
        }
        return deriv;
    }
#endif

    virtual void set_param(const std::vector<float>& new_param) override {
        if(new_param.size() != couplings.size())
            throw string("attempting to change size of couplings vector on set_param");
        copy(begin(new_param),end(new_param), begin(couplings));
    }
};
static RegisterNodeType<LinearCoupling,1> linear_coupling_node1("linear_coupling_uniform");
static RegisterNodeType<LinearCoupling,2> linear_coupling_node2("linear_coupling_with_inactivation");


struct NonlinearCoupling : public PotentialNode {
    CoordNode& input;
    int n_restype, n_coeff, n_res;
    float spline_offset, spline_inv_dx;
    vector<float> coeff;      // length n_restype*n_coeff
    vector<int>   coupling_types; // length input n_elem
    vector<float> weights;
    vector<float> wnumber;
    int num_independent_weight;

    NonlinearCoupling(hid_t grp, CoordNode& input_):
        PotentialNode(),
        input(input_),
        n_restype(get_dset_size(2,grp,"coeff")[0]),
        n_coeff  (get_dset_size(2,grp,"coeff")[1]),
        n_res(input.n_elem/n_restype),
        spline_offset(read_attribute<float>(grp,"coeff","spline_offset")),
        spline_inv_dx(read_attribute<float>(grp,"coeff","spline_inv_dx")),
        coeff(n_restype*n_coeff),
        coupling_types(n_res),
        weights(n_restype*n_restype),
        wnumber(n_res),
        num_independent_weight(read_attribute<int>(grp, ".", "number_independent_weights"))
    {
        check_elem_width(input, 1);  // could be generalized

        check_size(grp, "coupling_types", n_res);
        check_size(grp, "weights", n_restype*n_restype);

        traverse_dset<2,float>(grp,"coeff",[&](size_t nt, size_t nc, float x){coeff[nt*n_coeff+nc]=x;});
        traverse_dset<1,int>  (grp,"coupling_types",[&](size_t ne, int x){coupling_types[ne]=x;});
        traverse_dset<1,float>(grp,"weights",[&](size_t nw, float x){weights[nw]=x;});

        for(int i: coupling_types) if(i<0 || i>=n_restype) throw string("invalid coupling type");

        if (num_independent_weight != 1 and num_independent_weight != 20 and num_independent_weight != 400) 
            throw string("the number of independent weights should be 1, 20 or 400");

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("nonlinear_coupling", {n_res}, [&](float* buffer) {
                for(int nr: range(n_res)) { //  loop over residues
                    int ctype = coupling_types[nr]; // get CB type of current residue
                    wnumber[nr] = 0.0;
                    for(int aa: range(n_restype)) { // loop over types of SC beads that may fall within hemisphere
                        int nb = nr*n_restype+aa; // spans the counts of each SC bead type within every residue's CB hemisphere
                        wnumber[nr] += weights[ctype*n_restype+aa] * input.output(0, nb);
                    }
                    auto coord = (wnumber[nr]-spline_offset)*spline_inv_dx;
                    buffer[nr] = clamped_deBoor_value_and_deriv(
                            coeff.data() + ctype*n_coeff, coord, n_coeff)[0];
                }});
        }
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("nonlinear_coupling");

        float pot = 0.f;
        for(int nr: range(n_res)) {
            wnumber[nr] = 0.0;
            int ctype = coupling_types[nr];
            for(int aa: range(n_restype)) 
                wnumber[nr] += weights[ctype*n_restype+aa] * input.output(0, nr*n_restype+aa);

            auto coord = (wnumber[nr]-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(coeff.data() + ctype*n_coeff, coord, n_coeff);

            pot += v[0];
            for(int aa: range(n_restype)) 
                input.sens(0, nr*n_restype+aa) = weights[ctype*n_restype+aa] * spline_inv_dx * v[1];
        }
        
        potential = pot;
    }

    virtual std::vector<float> get_param() const override {
        int csize = coeff.size();
        int wsize = n_restype*n_restype;
        vector<float> params(csize+wsize, 0.f);

        for(int ct: range(csize)) 
            params[ct] = coeff[ct];

        for(int ct: range(wsize)) 
            params[ct+csize] = weights[ct];

        return params;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {

        int csize = coeff.size();
        int wsize = n_restype*n_restype;

        vector<float> deriv(csize+wsize, 0.f);
        vector<float> sens(n_res, 0.f);

        for(int nr: range(n_res)) {
            int starting_bin;
            float result[4];
            wnumber[nr] = 0.0;
            int ctype = coupling_types[nr];

            for(int aa: range(n_restype)) 
                wnumber[nr] += weights[ctype*n_restype+aa] * input.output(0, nr*n_restype+aa);

            auto coord = (wnumber[nr]-spline_offset)*spline_inv_dx;

            clamped_deBoor_coeff_deriv(&starting_bin, result, coord, n_coeff);
            for(int i: range(4)) deriv[ctype*n_coeff+starting_bin+i] += result[i];

            if ( num_independent_weight > 1) {
                auto v = clamped_deBoor_value_and_deriv(coeff.data() + ctype*n_coeff, coord, n_coeff);
                sens[nr] = spline_inv_dx * v[1];
            }
        }

        if ( num_independent_weight == 1 )
            return deriv;

        for(int nr: range(n_res)) {
            int ctype = coupling_types[nr];
            for(int aa: range(n_restype)) {
                auto deriv_aa = sens[nr] * input.output(0, nr*n_restype+aa); // dE/dw = dE/dn' * dn'/dw = sens*n, where n' = wn
                if ( num_independent_weight == 400)
                    deriv[csize+ctype*n_restype+aa] += deriv_aa;
                else if ( num_independent_weight == 20 ) 
                    for (int aa2: range(n_restype)) 
                        deriv[csize+aa2*n_restype+aa] += deriv_aa;
            }
        }
        
        return deriv;
    }
#endif

    virtual void set_param(const std::vector<float>& new_param) override {
        int csize = coeff.size();
        int wsize = n_restype*n_restype;
        if(new_param.size() != size_t(csize+wsize))
            throw string("the size of parameters should be the size of coeff + (the number of amino acid type)^2");

        std::vector<float> params(csize+wsize);
        copy(begin(new_param),end(new_param), begin(params));

        for( int i : range(csize)) coeff[i] = params[i];
        for(int aa: range(wsize)) weights[aa] = params[csize+aa];
    }
};
static RegisterNodeType<NonlinearCoupling,1> nonlinear_coupling_node("nonlinear_coupling");

struct InterNonlinearCoupling : public PotentialNode {
    CoordNode& input;
    int n_restype, n_coeff, n_res;
    float spline_offset, spline_inv_dx;
    vector<float> coeff;      // length n_restype*n_coeff
    vector<int>   coupling_types; // length input n_elem
    vector<float> weights;
    vector<float> wnumber;
    int num_independent_weight;

    InterNonlinearCoupling(hid_t grp, CoordNode& input_):
        PotentialNode(),
        input(input_),
        n_restype(get_dset_size(2,grp,"coeff")[0]),
        n_coeff  (get_dset_size(2,grp,"coeff")[1]),
        n_res(input.n_elem/n_restype),
        spline_offset(read_attribute<float>(grp,"coeff","spline_offset")),
        spline_inv_dx(read_attribute<float>(grp,"coeff","spline_inv_dx")),
        coeff(n_restype*n_coeff),
        coupling_types(n_res),
        weights(n_restype*n_restype),
        wnumber(n_res),
        num_independent_weight(read_attribute<int>(grp, ".", "number_independent_weights"))
    {
        check_elem_width(input, 2); // could be generalized

        check_size(grp, "coupling_types", n_res);
        check_size(grp, "weights", n_restype*n_restype);

        traverse_dset<2,float>(grp,"coeff",[&](size_t nt, size_t nc, float x){coeff[nt*n_coeff+nc]=x;});
        traverse_dset<1,int>  (grp,"coupling_types",[&](size_t ne, int x){coupling_types[ne]=x;});
        traverse_dset<1,float>(grp,"weights",[&](size_t nw, float x){weights[nw]=x;});

        for(int i: coupling_types) if(i<0 || i>=n_restype) throw string("invalid coupling type");

        if (num_independent_weight != 1 and num_independent_weight != 20 and num_independent_weight != 400) 
            throw string("the number of independent weights should be 1, 20 or 400");

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("inter_nonlinear_coupling", {n_res}, [&](float* buffer) {
                // printf("buffer[nr]\n");
                for(int nr: range(n_res)) { //  loop over residues
                    if (input.output(1, nr) > 0) {
                        int ctype = coupling_types[nr]; // get CB type of current residue
                        wnumber[nr] = 0.0;
                        for(int aa: range(n_restype)) { // loop over types of SC beads that may fall within hemisphere
                            int nb = nr*n_restype+aa; // spans the counts of each SC bead type within every residue's CB hemisphere
                            wnumber[nr] += weights[ctype*n_restype+aa] * input.output(0, nb);
                        }
                        auto coord = (wnumber[nr]-spline_offset)*spline_inv_dx;
                        buffer[nr] = clamped_deBoor_value_and_deriv(
                                coeff.data() + ctype*n_coeff, coord, n_coeff)[0];
                        // printf("%d: %.3f\n", nr, buffer[nr]);
                    }
                }});
        }
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("inter_nonlinear_coupling");

        float pot = 0.f;
        for(int nr: range(n_res)) {
            if (input.output(1, nr) > 0) {
                wnumber[nr] = 0.0;
                int ctype = coupling_types[nr];
                for(int aa: range(n_restype)) 
                    wnumber[nr] += weights[ctype*n_restype+aa] * input.output(0, nr*n_restype+aa);

                auto coord = (wnumber[nr]-spline_offset)*spline_inv_dx;
                auto v = clamped_deBoor_value_and_deriv(coeff.data() + ctype*n_coeff, coord, n_coeff);

                pot += v[0];
                for(int aa: range(n_restype)) 
                    input.sens(0, nr*n_restype+aa) = weights[ctype*n_restype+aa] * spline_inv_dx * v[1];
            } else {
                for(int aa: range(n_restype)) 
                    input.sens(0, nr*n_restype+aa) = 0.;
            }
        }
        
        potential = pot;
    }

    virtual std::vector<float> get_param() const override {
        int csize = coeff.size();
        int wsize = n_restype*n_restype;
        vector<float> params(csize+wsize, 0.f);

        for(int ct: range(csize)) 
            params[ct] = coeff[ct];

        for(int ct: range(wsize)) 
            params[ct+csize] = weights[ct];

        return params;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {

        int csize = coeff.size();
        int wsize = n_restype*n_restype;

        vector<float> deriv(csize+wsize, 0.f);
        vector<float> sens(n_res, 0.f);

        for(int nr: range(n_res)) {
            if (input.output(1, nr) > 0) {
                int starting_bin;
                float result[4];
                wnumber[nr] = 0.0;
                int ctype = coupling_types[nr];

                for(int aa: range(n_restype)) 
                    wnumber[nr] += weights[ctype*n_restype+aa] * input.output(0, nr*n_restype+aa);

                auto coord = (wnumber[nr]-spline_offset)*spline_inv_dx;

                clamped_deBoor_coeff_deriv(&starting_bin, result, coord, n_coeff);
                for(int i: range(4)) deriv[ctype*n_coeff+starting_bin+i] += result[i];

                if ( num_independent_weight > 1) {
                    auto v = clamped_deBoor_value_and_deriv(coeff.data() + ctype*n_coeff, coord, n_coeff);
                    sens[nr] = spline_inv_dx * v[1];
                }
            }
        }

        if ( num_independent_weight == 1 )
            return deriv;

        for(int nr: range(n_res)) {
            if (input.output(1, nr) > 0) {
                int ctype = coupling_types[nr];
                for(int aa: range(n_restype)) {
                    auto deriv_aa = sens[nr] * input.output(0, nr*n_restype+aa); // dE/dw = dE/dn' * dn'/dw = sens*n, where n' = wn
                    if ( num_independent_weight == 400)
                        deriv[csize+ctype*n_restype+aa] += deriv_aa;
                    else if ( num_independent_weight == 20 ) 
                        for (int aa2: range(n_restype)) 
                            deriv[csize+aa2*n_restype+aa] += deriv_aa;
                }
            }
        }
        
        return deriv;
    }
#endif

    virtual void set_param(const std::vector<float>& new_param) override {
        int csize = coeff.size();
        int wsize = n_restype*n_restype;
        if(new_param.size() != size_t(csize+wsize))
            throw string("the size of parameters should be the size of coeff + (the number of amino acid type)^2");

        std::vector<float> params(csize+wsize);
        copy(begin(new_param),end(new_param), begin(params));

        for( int i : range(csize)) coeff[i] = params[i];
        for(int aa: range(wsize)) weights[aa] = params[csize+aa];
    }
};
static RegisterNodeType<InterNonlinearCoupling,1> inter_nonlinear_coupling_node("inter_nonlinear_coupling");

}
