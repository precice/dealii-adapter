#ifndef TIME_H
#define TIME_H

#include <deal.II/base/timer.h>

namespace Adapter
{
  using namespace dealii;

  class Time
  {
  public:
    Time(const double time_end, const double delta_t)
      : timestep(0)
      , time_current(0.0)
      , time_end(time_end)
      , delta_t(delta_t)
    {}

    virtual ~Time()
    {}

    double
    current() const
    {
      return time_current;
    }

    double
    end() const
    {
      return time_end;
    }

    double
    get_delta_t() const
    {
      return delta_t;
    }

    unsigned int
    get_timestep() const
    {
      return timestep;
    }

    /**
     * @brief     set_absolute_time Allows to set the absolute time manually
     *
     * @param[in] new_time absolute time value
     */
    void
    set_absolute_time(const double new_time)
    {
      timestep     = new_time / delta_t;
      time_current = new_time;
    }

    void
    increment()
    {
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
  };
} // namespace Adapter
#endif // TIME_H
