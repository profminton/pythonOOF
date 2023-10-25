typedef struct simulation_type {
    double *doublevar_data;
    int doublevar_shape[2];
}c_simulation_type;
extern struct simulation_type* bind_simulation_init(int gridsize);
extern void bind_simulation_final(struct simulation_type *obj);
extern void bind_c2f_string(char* c_string, char* f_string);
extern void bind_f2c_string(char* f_string, char* c_string);