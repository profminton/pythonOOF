typedef struct simulation_type {
    double *elev_data;
    int elev_shape[2];
}c_simulation_type;
double c_double;
extern struct simulation_type* bind_simulation_init(int gridsize);
extern void bind_simulation_final(struct simulation_type *obj);