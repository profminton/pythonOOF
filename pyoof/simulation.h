typedef struct simulation_type {
    double *doublevar;
    int doublevar_shape[2];
    char *stringvar;
    int stringvar_len;
}c_simulation_type;
extern struct simulation_type* bind_simulation_init(int ny, int nx);
extern void bind_simulation_final(struct simulation_type *obj);
extern void bind_simulation_set_stringvar(struct simulation_type *obj, char *c_string);
extern char* bind_simulation_get_stringvar(struct simulation_type *obj);
