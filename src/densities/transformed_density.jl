# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    trafoof(d::Transformed)::AbstractMeasure

Get the transform from `parent(d)` to `d`, so that

```julia
trafoof(d)(parent(d)) == d
```
"""
function trafoof end
export trafoof


abstract type TDVolCorr end
struct TDNoCorr <: TDVolCorr end
struct TDLADJCorr <: TDVolCorr end


"""
    Transformed

*BAT-internal, not part of stable public API.*
"""
struct Transformed{D<:AbstractMeasure,FT<:Function,VC<:TDVolCorr,VS<:AbstractValueShape} <: AbstractMeasure
    orig::D
    trafo::FT  # ToDo: store inverse(trafo) instead?
    volcorr::VC
    _varshape::VS
end

function Transformed(orig::D, trafo::Function, volcorr::TDVolCorr) where D<:AbstractMeasure
    vs = trafo(varshape(orig))
    Transformed(orig, trafo, volcorr, vs)
end


@inline function (trafo::DistributionTransform)(density::AbstractMeasure; volcorr::Val{vc} = Val(true)) where vc
    if vc
        Transformed(density, trafo, TDLADJCorr())
    else
        Transformed(density, trafo, TDNoCorr())
    end
end


Base.parent(density::Transformed) = density.orig
trafoof(density::Transformed) = density.trafo

@inline DensityInterface.DensityKind(x::Transformed) = DensityKind(x.orig)

ValueShapes.varshape(density::Transformed) = density._varshape

# ToDo: Should not be neccessary, improve default implementation of
# ValueShapes.totalndof(density::AbstractMeasure):
ValueShapes.totalndof(density::Transformed) = totalndof(varshape(density))

var_bounds(density::Transformed{<:Any,<:DistributionTransform}) = dist_param_bounds(density.trafo.target_dist)


function DensityInterface.logdensityof(density::Transformed{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inverse(density.trafo)(v)
    logdensityof(parent(density), v_orig)
end

function checked_logdensityof(density::Transformed{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inverse(density.trafo)(v)
    checked_logdensityof(parent(density), v_orig)
end


function _v_orig_and_ladj(density::Transformed, v::Any)
    with_logabsdet_jacobian(inverse(density.trafo), v)
end

function DensityInterface.logdensityof(density::Transformed{D,FT,TDLADJCorr}, v::Any) where {D,FT,}
    v_orig, ladj = _v_orig_and_ladj(density, v)
    logd_orig = logdensityof(parent(density), v_orig)
    _combine_logd_with_ladj(logd_orig, ladj)
end

function checked_logdensityof(density::Transformed{D,FT,TDLADJCorr}, v::Any) where {D,FT,}
    v_orig, ladj = _v_orig_and_ladj(density, v)
    logd_orig = logdensityof(parent(density), v_orig)
    isnan(logd_orig) && @throw_logged EvalException(logdensityof, density, v, 0)
    _combine_logd_with_ladj(logd_orig, ladj)
end
