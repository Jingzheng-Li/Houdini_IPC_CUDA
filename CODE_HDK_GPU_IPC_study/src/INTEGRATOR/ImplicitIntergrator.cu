
#include "ImplicitIntergrator.cuh"

// TODO: split computeXTilta here

ImplicitIntegrator::ImplicitIntegrator(std::unique_ptr<GeometryManager>& instance) 
    : m_instance(instance) {}

ImplicitIntegrator::~ImplicitIntegrator() {};

