# Production Planning

## Purpose

Defines how production batches are scheduled, balancing customer
demand against material availability, so production runs efficiently
without stockouts or excess inventory.

## Scope

Applies to all scheduled production batches across product lines.

## Planning Cycle

Production is planned on a **weekly** basis — each week's batches are
scheduled in advance rather than decided day-to-day, giving enough
lead time to confirm materials are available before a batch is
committed.

## Planning Inputs

Each week's schedule is built from two things together, not either
alone:

- **Demand** — expected sales orders and forecast for the upcoming
  period
- **Material availability** — current raw material stock levels and
  any pending deliveries (cross-checked against the Raw Material
  Reorder Policy and Stock Receiving Procedure)

A batch is only scheduled once both demand and material availability
support it — high demand for a product doesn't get scheduled if the
raw materials aren't confirmed available in time.

## Decision Process

The weekly production schedule is set by the **Production Lead**, with
required input from **Inventory** before it's finalized. Inventory
confirms material availability for the proposed schedule; if a
material shortfall is flagged, the schedule is adjusted (reordered
priority, delayed batch, or substituted timing) before it's approved.

## Responsibilities

- **Production Lead**: proposes and finalizes the weekly schedule
- **Inventory Officer**: confirms material availability, flags
  shortfalls before the schedule is locked in
